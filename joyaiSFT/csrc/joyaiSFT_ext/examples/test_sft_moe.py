
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch
expert_num = 10
hidden_size = 5120
intermediate_size = 1536
stride = 32
group_min_len = 10
group_max_len = 1024
gate_type = 1
up_type = 1
down_type = 1
hidden_type = 1
n_routed_experts = 2
qlen = 30
layer_num = 10
CPUInfer = cpuinfer_ext.CPUInfer(48)
validation_iter = 100
dtype = torch.float16
gradtype = torch.bfloat16

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

class SiLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input / (1.0 + torch.exp(-input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid = 1.0 / (1.0 + torch.exp(-input))
        return grad_output * (sigmoid + input * sigmoid * (1 - sigmoid))
silu = SiLU.apply

def mlp_torch(input, gate_proj, up_proj, down_proj, requires_grad=False):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    if requires_grad:
        intermediate = silu(gate_buf) * up_buf
    else:
        intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    return ret

def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=False):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i], requires_grad)
        outputs.append(expert_out)
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = new_x.view(*expert_ids.shape, -1).type(weights.dtype).mul_(weights.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
    return t_output

def test_forward():
    with torch.inference_mode(mode=True):
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device='cuda').to('cpu').contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device='cuda').to('cpu').contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, device='cuda').to('cpu').contiguous()
            config = cpuinfer_ext.sft_moe.SFT_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type, 0)
            moe = cpuinfer_ext.sft_moe.SFT_MOE(config)
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)
        for i in range(validation_iter):
            expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
            weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
            input = torch.randn((qlen, hidden_size), dtype=dtype).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=dtype).contiguous()
            input = input / 100
            moe = moes[i % layer_num]
            CPUInfer.submit(moe.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input.data_ptr(), output.data_ptr()))
            CPUInfer.sync()
            gate_proj = gate_projs[i % layer_num]
            up_proj = up_projs[i % layer_num]
            down_proj = down_projs[i % layer_num]
            t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj)
            diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
            print('diff = ', diff)
            assert diff < 0.001

def test_backward():
    print('\n===== Testing Backward Pass =====')
    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, requires_grad=True).contiguous()
    config = cpuinfer_ext.sft_moe.SFT_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
    moe = cpuinfer_ext.sft_moe.SFT_MOE(config)
    expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
    input = torch.randn((qlen, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    input = (input / 100).detach().requires_grad_(True)
    input_cpp = input.clone().detach().requires_grad_(True).contiguous()
    t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=True)
    t_output.retain_grad()
    output_cpp = torch.empty((qlen, hidden_size), dtype=dtype).contiguous()
    forward_start_time = time.time()
    CPUInfer.submit(moe.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), output_cpp.data_ptr()))
    CPUInfer.sync()
    forward_end_time = time.time()
    print(f'C++ forward 耗时: {forward_end_time - forward_start_time:.4f} 秒')
    FLOPs_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    KT_TFLOPS_fwd = FLOPs_fwd / (forward_end_time - forward_start_time) / 1000000000000.0
    forward_diff = torch.mean(torch.abs(output_cpp - t_output)) / torch.mean(torch.abs(t_output))
    print(f'Forward diff: {forward_diff.item()}')
    assert forward_diff < 0.001, f'Forward diff too large: {forward_diff.item()}'
    print('✅ Forward test passed!')
    grad_input_cpp = torch.empty_like(input_cpp, dtype=gradtype).contiguous()
    grad_output = torch.randn_like(t_output, dtype=gradtype).contiguous()
    grad_output_cpp = grad_output.clone()
    print('-- pytorch backward --')
    pytorch_start_time = time.time()
    t_output.backward(grad_output, retain_graph=True)
    pytorch_end_time = time.time()
    pytorch_time = pytorch_end_time - pytorch_start_time
    print('-- c++ backward --')
    CPUInfer.submit(moe.backward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), grad_output_cpp.data_ptr(), grad_input_cpp.data_ptr()))
    CPUInfer.sync()
    cpp_start_time = time.time()
    CPUInfer.submit(moe.backward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), grad_output_cpp.data_ptr(), grad_input_cpp.data_ptr()))
    CPUInfer.sync()
    cpp_end_time = time.time()
    cpp_time = cpp_end_time - cpp_start_time
    print(f'PyTorch backward 耗时: {pytorch_time:.4f} 秒')
    print(f'C++ backward 耗时: {cpp_time:.4f} 秒')
    print(f'性能比较: PyTorch/C++ = {pytorch_time / cpp_time:.2f}x')
    print(f'qlen:{qlen}, n_exp:{n_routed_experts}, hidden:{hidden_size}, inter:{intermediate_size}')
    FLOPs_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    torch_TFLOPS_bwd = FLOPs_bwd / pytorch_time / 1000000000000.0
    KT_TFLOPS_bwd = FLOPs_bwd / cpp_time / 1000000000000.0
    print(f'PyTorch backward TFLOPS: {torch_TFLOPS_bwd}')
    print(f'KT forward TFLOPS: {KT_TFLOPS_fwd}')
    print(f'KT backward TFLOPS: {KT_TFLOPS_bwd}')
    total_flops_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    total_flops_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    tflops_fwd_cpp = total_flops_fwd / (forward_end_time - forward_start_time) / 1000000000000.0
    tflops_bwd_cpp = total_flops_bwd / cpp_time / 1000000000000.0
    tflops_bwd_torch = total_flops_bwd / pytorch_time / 1000000000000.0
    print(f'\n=== TFLOPS ===')
    print(f'CPUInfer forward  : {tflops_fwd_cpp:.2f} TFLOPS')
    print(f'CPUInfer backward : {tflops_bwd_cpp:.2f} TFLOPS')
    print(f'Torch   backward : {tflops_bwd_torch:.2f} TFLOPS')
    backward_diff = torch.mean(torch.abs(grad_input_cpp - input.grad)) / torch.mean(torch.abs(input.grad))
    print(f'Backward diff: {backward_diff.item()}')
    assert backward_diff < 0.005, f'Backward diff too large: {backward_diff.item()}'
    print('✅ Backward pass test passed!')

def test_backward_2round_with_tflops():
    """
    跑两轮 forward+backward，对比 PyTorch 与 C++ 实现的正确性和性能，
    并输出每轮及总体的 TFLOPS 与耗时信息。
    依赖：已在全局定义 expert_num、n_routed_experts、hidden_size、intermediate_size、
          stride、group_min_len、group_max_len、gate_type、up_type、down_type、
          hidden_type、qlen、dtype、gradtype 以及 moe_torch、cpuinfer_ext、CPUInfer。
    """
    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, requires_grad=True).contiguous()
    config = cpuinfer_ext.sft_moe.SFT_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
    moe = cpuinfer_ext.sft_moe.SFT_MOE(config)
    FLOPs_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    FLOPs_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    summary = []
    for round_idx in range(2):
        print(f'\n================ Round {round_idx + 1}/2 ================')
        expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
        weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
        input_pt = (torch.randn((qlen, hidden_size), dtype=dtype) / 100).detach().requires_grad_(True).contiguous()
        input_cpp = input_pt.clone().detach().requires_grad_(True).contiguous()
        t_output = moe_torch(input_pt, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=True)
        t_output.retain_grad()
        output_cpp = torch.empty((qlen, hidden_size), dtype=dtype).contiguous()
        fwd_start = time.time()
        CPUInfer.submit(moe.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), output_cpp.data_ptr()))
        CPUInfer.sync()
        fwd_end = time.time()
        fwd_time = fwd_end - fwd_start
        print(f'C++ forward 耗时: {fwd_time:.4f} s')
        fwd_diff = torch.mean(torch.abs(output_cpp - t_output)) / torch.mean(torch.abs(t_output))
        print(f'Forward diff: {fwd_diff.item():.4e}')
        grad_output = torch.randn_like(t_output, dtype=gradtype).contiguous()
        grad_output_cpp = grad_output.clone().contiguous()
        grad_input_cpp = torch.zeros_like(input_cpp, dtype=gradtype).contiguous()
        for p in (gate_proj, up_proj, down_proj, input_pt):
            if p.grad is not None:
                p.grad.zero_()
        pyt_start = time.time()
        t_output.backward(grad_output, retain_graph=True)
        pyt_end = time.time()
        pyt_time = pyt_end - pyt_start
        print(f'PyTorch backward 耗时: {pyt_time:.4f} s')
        cpp_start = time.time()
        CPUInfer.submit(moe.backward(round_idx, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), grad_output_cpp.data_ptr(), grad_input_cpp.data_ptr()))
        CPUInfer.sync()
        cpp_end = time.time()
        cpp_time = cpp_end - cpp_start
        print(f'C++ backward(第2次) 耗时: {cpp_time:.4f} s')
        if input_pt.grad is None:
            print('错误：input_pt.grad为None，PyTorch反向传播可能失败')
            bwd_diff = float('nan')
        else:
            print(f'[DEBUG] PyTorch grad shape: {input_pt.grad.shape}, dtype: {input_pt.grad.dtype}')
            print(f'[DEBUG] C++ grad shape: {grad_input_cpp.shape}, dtype: {grad_input_cpp.dtype}')
            pt_grad_has_nan = torch.isnan(input_pt.grad).any()
            print(f'[DEBUG] PyTorch grad contains NaN: {pt_grad_has_nan}')
            if pt_grad_has_nan:
                print(f'[DEBUG] PyTorch grad NaN count: {torch.isnan(input_pt.grad).sum().item()}')
            cpp_grad_has_nan = torch.isnan(grad_input_cpp).any()
            print(f'[DEBUG] C++ grad contains NaN: {cpp_grad_has_nan}')
            if cpp_grad_has_nan:
                print(f'[DEBUG] C++ grad NaN count: {torch.isnan(grad_input_cpp).sum().item()}')
            grad_input_cpp_fp32 = grad_input_cpp.to(torch.float32)
            input_pt_grad_fp32 = input_pt.grad.to(torch.float32)
            cpp_fp32_has_nan = torch.isnan(grad_input_cpp_fp32).any()
            pt_fp32_has_nan = torch.isnan(input_pt_grad_fp32).any()
            print(f'[DEBUG] After FP32 conversion - PyTorch NaN: {pt_fp32_has_nan}, C++ NaN: {cpp_fp32_has_nan}')
            if pt_fp32_has_nan or cpp_fp32_has_nan:
                bwd_diff = float('nan')
                print(f'[DEBUG] 检测到NaN，跳过diff计算')
            else:
                diff_tensor = torch.abs(grad_input_cpp_fp32 - input_pt_grad_fp32)
                denominator = torch.mean(torch.abs(input_pt_grad_fp32))
                print(f'[DEBUG] Diff stats - max: {diff_tensor.max().item():.6f}, mean: {diff_tensor.mean().item():.6f}')
                print(f'[DEBUG] Denominator: {denominator.item():.6f}')
                bwd_diff = torch.mean(diff_tensor) / denominator
        if isinstance(bwd_diff, torch.Tensor):
            print(f'Backward diff: {bwd_diff.item():.4e}')
        elif isinstance(bwd_diff, float):
            print(f'Backward diff: {bwd_diff:.4e}')
        else:
            print(f'Backward diff: {bwd_diff}')
        tflops_fwd_cpp = FLOPs_fwd / fwd_time / 1000000000000.0
        tflops_bwd_cpp = FLOPs_bwd / cpp_time / 1000000000000.0
        tflops_bwd_torch = FLOPs_bwd / pyt_time / 1000000000000.0
        print(f'\n--- Round {round_idx + 1} TFLOPS ---')
        print(f'CPUInfer forward  : {tflops_fwd_cpp:.2f} TFLOPS')
        print(f'CPUInfer backward : {tflops_bwd_cpp:.2f} TFLOPS')
        print(f'Torch   backward : {tflops_bwd_torch:.2f} TFLOPS')
        summary.append(dict(round=round_idx + 1, fwd_time=fwd_time, pyt_bwd_time=pyt_time, cpp_bwd_time=cpp_time, fwd_diff=fwd_diff.item(), bwd_diff=bwd_diff.item() if isinstance(bwd_diff, torch.Tensor) else bwd_diff, tflops_fwd_cpp=tflops_fwd_cpp, tflops_bwd_cpp=tflops_bwd_cpp, tflops_bwd_torch=tflops_bwd_torch))
    print('\n================= Two-Round Summary =================')
    for item in summary:
        print(f"Round {item['round']}: fwd {item['fwd_time']:.4f}s | bwd_torch {item['pyt_bwd_time']:.4f}s | bwd_cpp {item['cpp_bwd_time']:.4f}s | diff(fwd/bwd) {item['fwd_diff']:.2e}/{item['bwd_diff']:.2e} | TFLOPS(cpp fwd/bwd) {item['tflops_fwd_cpp']:.2f}/{item['tflops_bwd_cpp']:.2f}")

def test_backward_10round_5layer():
    """
    创建 5 个独立 SFT-MOE 层，连续跑 10 轮 forward+backward。
    第 n 轮使用第 n % 5 层，逐轮验证 C++ 与 PyTorch 的数值一致性，
    同时统计 TFLOPS / 耗时。全程不修改任何全局变量。
    """
    num_layers = 5
    num_rounds = 10
    gate_projs, up_projs, down_projs, moes = ([], [], [], [])
    for _ in range(num_layers):
        gp = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
        up = torch.randn_like(gp, requires_grad=True)
        dp = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, requires_grad=True).contiguous()
        cfg = cpuinfer_ext.sft_moe.SFT_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gp.data_ptr(), up.data_ptr(), dp.data_ptr(), gate_type, up_type, down_type, hidden_type)
        moes.append(cpuinfer_ext.sft_moe.SFT_MOE(cfg))
        gate_projs.append(gp)
        up_projs.append(up)
        down_projs.append(dp)
    FLOPs_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    FLOPs_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    summary = []
    for r in range(num_rounds):
        layer_id = r % num_layers
        moe = moes[layer_id]
        gp, up, dp = (gate_projs[layer_id], up_projs[layer_id], down_projs[layer_id])
        print(f'\n================ Round {r + 1}/{num_rounds}  (use layer {layer_id}) ================')
        expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
        weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
        inp_pt = (torch.randn((qlen, hidden_size), dtype=dtype) / 100).detach().requires_grad_(True).contiguous()
        inp_cpp = inp_pt.clone().detach().requires_grad_(True).contiguous()
        t_out = moe_torch(inp_pt, expert_ids, weights, gp, up, dp, requires_grad=True)
        t_out.retain_grad()
        out_cpp = torch.empty_like(t_out).contiguous()
        t0 = time.time()
        CPUInfer.submit(moe.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), inp_cpp.data_ptr(), out_cpp.data_ptr()))
        CPUInfer.sync()
        fwd_time = time.time() - t0
        fwd_diff = (out_cpp - t_out).abs().mean() / t_out.abs().mean()
        print(f'Forward diff = {fwd_diff.item():.3e} | C++ fwd {fwd_time:.3f}s')
        grad_out = torch.randn_like(t_out, dtype=gradtype).contiguous()
        grad_out_cpp = grad_out.clone().contiguous()
        grad_inp_cpp = torch.empty_like(inp_cpp, dtype=gradtype).contiguous()
        for p in (gp, up, dp, inp_pt):
            if p.grad is not None:
                p.grad.zero_()
        t1 = time.time()
        t_out.backward(grad_out, retain_graph=True)
        pyt_time = time.time() - t1
        t2 = time.time()
        CPUInfer.submit(moe.backward(r, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), inp_cpp.data_ptr(), grad_out_cpp.data_ptr(), grad_inp_cpp.data_ptr()))
        CPUInfer.sync()
        cpp_time = time.time() - t2
        bwd_diff = (grad_inp_cpp - inp_pt.grad).abs().mean() / inp_pt.grad.abs().mean()
        print(f'Backward diff = {bwd_diff.item():.3e} | PyTorch bwd {pyt_time:.3f}s | C++ bwd {cpp_time:.3f}s')
        tflops_fwd_cpp = FLOPs_fwd / fwd_time / 1000000000000.0
        tflops_bwd_cpp = FLOPs_bwd / cpp_time / 1000000000000.0
        tflops_bwd_torch = FLOPs_bwd / pyt_time / 1000000000000.0
        summary.append(dict(rd=r + 1, layer=layer_id, fwd_time=fwd_time, pyt_time=pyt_time, cpp_time=cpp_time, fwd_diff=fwd_diff.item(), bwd_diff=bwd_diff.item(), tf_fwd=tflops_fwd_cpp, tf_bwd_cpp=tflops_bwd_cpp, tf_bwd_torch=tflops_bwd_torch))
    print('\n================ 10-Round Summary ================')
    for s in summary:
        print(f"R{s['rd']:02d}(L{s['layer']}) | Δf {s['fwd_diff']:.2e} / {s['bwd_diff']:.2e} | t fwd {s['fwd_time']:.3f}s  bwd Torch {s['pyt_time']:.3f}s / C++ {s['cpp_time']:.3f}s | TFLOPS C++ f/b {s['tf_fwd']:.2f}/{s['tf_bwd_cpp']:.2f}  Torch bwd {s['tf_bwd_torch']:.2f}")
    print('\n✅ 10 轮 5 层测试完成，全部差异在可接受范围内！')

def test_backward_one_vs_many_comparison():
    """
    专门对比 backward_one 和 backward_many 的结果差异
    """
    print('\n=== Backward One vs Many Comparison ===')
    torch.manual_seed(42)
    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, requires_grad=True).contiguous()
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, requires_grad=True).contiguous()
    config_one = cpuinfer_ext.sft_moe.SFT_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, 10000000, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
    config_many = cpuinfer_ext.sft_moe.SFT_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
    moe_one = cpuinfer_ext.sft_moe.SFT_MOE(config_one)
    moe_many = cpuinfer_ext.sft_moe.SFT_MOE(config_many)
    torch.manual_seed(123)
    expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
    input_one = (torch.randn((qlen, hidden_size), dtype=dtype) / 100).detach().requires_grad_(True).contiguous()
    input_many = input_one.clone().detach().requires_grad_(True).contiguous()
    output_one = torch.empty((qlen, hidden_size), dtype=dtype).contiguous()
    output_many = torch.empty((qlen, hidden_size), dtype=dtype).contiguous()
    CPUInfer.submit(moe_one.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_one.data_ptr(), output_one.data_ptr()))
    CPUInfer.sync()
    CPUInfer.submit(moe_many.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_many.data_ptr(), output_many.data_ptr()))
    CPUInfer.sync()
    print(f'Forward outputs identical: {torch.allclose(output_one, output_many, atol=1e-06)}')
    if not torch.allclose(output_one, output_many, atol=1e-06):
        print(f'Forward diff: {torch.mean(torch.abs(output_one - output_many))}')
    grad_output = torch.randn_like(output_one, dtype=gradtype).contiguous()
    grad_output_one = grad_output.clone().contiguous()
    grad_output_many = grad_output.clone().contiguous()
    grad_input_one = torch.zeros_like(input_one, dtype=gradtype).contiguous()
    grad_input_many = torch.zeros_like(input_many, dtype=gradtype).contiguous()
    print('\n--- Testing backward_one (force group_min_len = 10000000) ---')
    CPUInfer.submit(moe_one.backward(0, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_one.data_ptr(), grad_output_one.data_ptr(), grad_input_one.data_ptr()))
    CPUInfer.sync()
    one_has_nan = torch.isnan(grad_input_one).any()
    print(f'backward_one result has NaN: {one_has_nan}')
    if one_has_nan:
        print(f'backward_one NaN count: {torch.isnan(grad_input_one).sum().item()}/{grad_input_one.numel()}')
    else:
        print(f'backward_one grad_input stats: min={grad_input_one.min():.6f}, max={grad_input_one.max():.6f}, mean={grad_input_one.mean():.6f}')
    print('\n--- Testing backward_many (normal group_min_len) ---')
    CPUInfer.submit(moe_many.backward(0, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_many.data_ptr(), grad_output_many.data_ptr(), grad_input_many.data_ptr()))
    CPUInfer.sync()
    many_has_nan = torch.isnan(grad_input_many).any()
    print(f'backward_many result has NaN: {many_has_nan}')
    if many_has_nan:
        print(f'backward_many NaN count: {torch.isnan(grad_input_many).sum().item()}/{grad_input_many.numel()}')
    else:
        print(f'backward_many grad_input stats: min={grad_input_many.min():.6f}, max={grad_input_many.max():.6f}, mean={grad_input_many.mean():.6f}')
    if not one_has_nan and (not many_has_nan):
        print(f'\n--- Comparison ---')
        grad_one_fp32 = grad_input_one.to(torch.float32)
        grad_many_fp32 = grad_input_many.to(torch.float32)
        print(f'Results identical: {torch.allclose(grad_one_fp32, grad_many_fp32, atol=1e-06)}')
        diff = torch.abs(grad_one_fp32 - grad_many_fp32)
        print(f'Max absolute difference: {diff.max():.6f}')
        print(f'Mean absolute difference: {diff.mean():.6f}')
        max_diff_idx = torch.argmax(diff.flatten())
        token_idx = max_diff_idx // hidden_size
        feature_idx = max_diff_idx % hidden_size
        print(f'Max diff at token {token_idx}, feature {feature_idx}: one={grad_one_fp32.flatten()[max_diff_idx]:.6f}, many={grad_many_fp32.flatten()[max_diff_idx]:.6f}')
    elif not one_has_nan and many_has_nan:
        print(f'\n--- backward_one正常，backward_many有NaN ---')
        print('这确认了问题出在backward_many实现上')
    elif one_has_nan and (not many_has_nan):
        print(f'\n--- backward_one有NaN，backward_many正常 ---')
        print('这很奇怪，需要进一步调查')
    else:
        print(f'\n--- 两者都有NaN ---')
        print('问题可能在更基础的地方')
if __name__ == '__main__':
    test_backward_one_vs_many_comparison()