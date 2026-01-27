import csv
import torch
import torch.nn as nn
import vLLMMarlin
torch.set_grad_enabled(False)
from utils.marlin_utils import MarlinWorkspace, marlin_quantize, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MIN_THREAD_K, GPTQ_MARLIN_MAX_PARALLEL

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
setup_seed(20241223)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)
global_dtype = torch.bfloat16
global_device = torch.device('cuda', 0)
global_num_cases: int = int(50)
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
max_batch_size = 512
max_tp = 8
L2_size = 73728 * 1024

def get_usable_mem():
    properties = torch.cuda.get_device_properties(global_device)
    allocated_memory = torch.cuda.memory_allocated(global_device)
    reserved_memory = torch.cuda.memory_reserved(global_device)
    return properties.total_memory - 512 * 1024 ** 2 - allocated_memory

def exp_range(start, stop, step=2):
    now = start
    while now <= stop:
        yield now
        now *= step

def timing(func, iters, epochs=100):
    for idx in range(iters):
        func(idx)
    torch.cuda.synchronize()
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        for idx in range(iters):
            func(idx)
    for _ in range(2000):
        cuda_graph.replay()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(10):
        cuda_graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms0 = start_event.elapsed_time(end_event)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(epochs + 10):
        cuda_graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) - elapsed_time_ms0
    return elapsed_time_ms / iters / epochs

class LinearMarlin(nn.Linear):
    marlin_q_w: torch.Tensor
    marlin_s: torch.Tensor
    g_idx: torch.Tensor
    sort_indices: torch.Tensor
    has_bias: bool

    def __init__(self, in_features, out_features, bias=False, device: str='cuda', num_bits: int=4, group_size: int=64, act_order: bool=False, is_k_full=True, sms=-1, **kwargs):
        self.padding = False
        assert device.lower() != 'cpu', 'Marlin quantized linear only supports GPU device'
        if in_features % GPTQ_MARLIN_MIN_THREAD_K != 0 or out_features % GPTQ_MARLIN_MIN_THREAD_K != 0:
            self.padding = True
            self.orin_in_features = in_features
            self.orin_out_features = out_features
            in_features = (in_features + GPTQ_MARLIN_MIN_THREAD_K - 1) // GPTQ_MARLIN_MIN_THREAD_K * GPTQ_MARLIN_MIN_THREAD_K
            out_features = (out_features + GPTQ_MARLIN_MIN_THREAD_N - 1) // GPTQ_MARLIN_MIN_THREAD_N * GPTQ_MARLIN_MIN_THREAD_N
        super().__init__(in_features, out_features, bias, device)
        self.has_bias = bias
        self.device = device
        self.num_bits = num_bits
        self.group_size = group_size
        self.act_order = act_order
        blocks_k, blocks_n = (in_features // 128, out_features // 128)
        self.sms = sms
        self.is_k_full = is_k_full
        self.weight.requires_grad = False
        self.weight.t_()
        marlin_q_w = torch.randint(int(-1000000000.0), int(1000000000.0), (in_features // 16, out_features * 2), device=device, dtype=torch.int)
        marlin_s = torch.randn((in_features // 64, out_features), device=device)
        self.workspace = MarlinWorkspace(self.out_features, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL, self.device)
        self.marlin_q_w = marlin_q_w
        self.marlin_s = marlin_s
        self.g_idx = torch.empty(0, dtype=torch.int32, device=self.device)
        self.sort_indices = torch.empty(0, dtype=torch.int32, device=self.device)
        self.k = self.weight.shape[0]
        self.n = self.weight.shape[1]
        self.weight = None
        '\n\t\tprint(in_features, out_features)\n\t\tprint(marlin_q_w.shape)\n\t\tprint(marlin_q_w.dtype)\n\t\tprint(marlin_s.shape)\n\t\tprint(marlin_s.dtype)\n\t\tprint(self.workspace.scratch.shape)\n\t\tprint(self.workspace.scratch.dtype)\n\t\tprint(self.g_idx.shape)\n\t\tprint(self.g_idx.dtype)\n\t\tprint(self.sort_indices.shape)\n\t\tprint(self.sort_indices.dtype)\n\t\t#print(w_ref.shape)\n\t\t#print(w_ref.dtype)\n\t\t'

    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        if self.padding:
            padding_input = torch.empty(x.shape[0], self.in_features, device=x.device, dtype=x.dtype)
            padding_input[:, :self.orin_in_features] = x
            x = padding_input
        marlin_s = self.marlin_s.to(x.dtype)
        sms = self.sms
        x = vLLMMarlin.gptq_marlin_gemm(x, self.marlin_q_w, marlin_s, self.g_idx, self.sort_indices, self.workspace.scratch, self.num_bits, bsz_tensor, x.shape[0], self.n, x.shape[-1], sms, self.is_k_full)
        if self.has_bias:
            x = x + self.bias
        if self.padding:
            x = x[:, :self.orin_out_features]
            orig_shape[-1] = self.orin_out_features
        else:
            orig_shape[-1] = self.out_features
        return x.reshape(orig_shape).to(orig_dtype)

def benchLinearMarlin(input_dim, output_dim):
    print('benchmarking MLP Marlin')
    print('-----------------------------------------------------------')
    headers = ['batch_size', 'tp', 'used_time', 'bandwidth GB/s', 'TFLOPS', 'cases', 'padding', 'sms']
    print(' | '.join(headers) + '\n')
    rows = []
    for batch_size in exp_range(1, 64):
        for tp in exp_range(1, max_tp):
            torch.cuda.empty_cache()
            if output_dim % tp != 0:
                continue
            cur_output_dim = output_dim // tp
            modules = []
            inputs = []
            data_size = int(0.53125 * input_dim * cur_output_dim)
            input_size = int(2 * batch_size * input_dim)
            output_size = int(2 * batch_size * cur_output_dim)
            usable_mem = get_usable_mem() - 2 * input_dim * cur_output_dim
            min_cases = max(global_num_cases, 2 * L2_size // (data_size + input_size))
            cases = int(min(min_cases, usable_mem * 0.8 // (data_size + input_size)))
            bsz_tensor = torch.tensor([batch_size], device=global_device, dtype=torch.int32)
            if cases == 0:
                row = [f'{batch_size}', 'OOM', 'OOM', 'OOM', '0', 'False']
                rows.append(row)
                break
            for _ in range(cases):
                modules.append(LinearMarlin(input_dim, cur_output_dim, sms=56, non_equal_division=False).to(device=global_device).eval())
                inputs.append(torch.randn(batch_size, 1, input_dim, device=global_device))

            def forward(case_id):
                modules[case_id](inputs[case_id], bsz_tensor)
            used_time = timing(forward, iters=cases)
            bandwidth = (data_size + input_size + output_size) / used_time / 1000000.0
            flops = 2 * batch_size * input_dim * cur_output_dim
            tflops = flops / used_time / 1000000000.0
            cur_sms = modules[0].sms
            row = [f'{batch_size}', f'{tp}', f'{used_time}', f'{bandwidth}', f'{tflops}', f'{cases}', modules[0].padding, cur_sms]
            rows.append(row)
            print(f'{batch_size}', f'{tp}', f'{used_time}', f'{bandwidth}', f'{tflops}', f'{cases}', modules[0].padding, cur_sms)
    "\n\twith open(out_file, 'w', newline='') as csvfile:\n\t\tcsvwriter = csv.writer(csvfile)\n\t\tcsvwriter.writerow(headers)\n\t\tfor row in rows:\n\t\t\tcsvwriter.writerow(row)\n\t"
    '\n\tmarkdown_table = " | ".join(headers) + "\n"\n\tmarkdown_table += " | ".join(["---"] * len(headers)) + "\n"\n\tfor row in rows:\n\t\tmarkdown_table += " | ".join(row) + "\n"\n\n\tprint(markdown_table)\n\t'
if __name__ == '__main__':
    benchLinearMarlin(5120, 3584)
    exit(0)
    max_batch = 1
    cur_batch = 1
    marlin_linear = LinearMarlin(5120, 3584)
    input_tensor = torch.randn(max_batch, 1, 5120, device='cuda', dtype=torch.bfloat16)
    bsz_tensor = torch.tensor([max_batch], device='cuda', dtype=torch.int32)
    out_truth = marlin_linear(input_tensor, bsz_tensor)
    print(out_truth)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf = marlin_linear(input_tensor, bsz_tensor)
    for i in range(10000):
        g.replay()
    marlin_linear = LinearMarlin(5120, 3584)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf = marlin_linear(input_tensor, bsz_tensor)
    new_input = torch.randn(cur_batch, 1, 5120, device='cuda', dtype=torch.bfloat16)
    bsz_tensor.copy_(torch.tensor([cur_batch], device='cuda', dtype=torch.int32))
    new_out_truth = marlin_linear(new_input, bsz_tensor)
    input_tensor[:cur_batch].copy_(new_input)
    input_tensor[cur_batch:] = 0
    g.replay()
    torch.cuda.synchronize()

    def printMinMax(tensor):
        abs_tensor = torch.abs(tensor)
        min_val = torch.min(abs_tensor)
        max_val = torch.max(abs_tensor)
        min_indices = (abs_tensor == min_val).nonzero(as_tuple=True)
        max_indices = (abs_tensor == max_val).nonzero(as_tuple=True)
        print(f'min: {min_val.item()}')
        print(f'min idx: {min_indices}')
        print(f'max: {max_val.item()}')
        print(f'max idx: {max_indices}')
    print(out_buf[:cur_batch].shape)
    print(new_out_truth.shape)
    printMinMax(out_buf[:cur_batch])
    printMinMax(new_out_truth)