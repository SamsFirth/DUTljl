import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch
expert_num = 160
hidden_size = 5120
intermediate_size = 1536
stride = 16
group_min_len = 10
group_max_len = 1024
n_routed_experts = 6
layer_num = 10
qlen = 1
CPUInfer = cpuinfer_ext.CPUInfer(64)
warm_up_iter = 1000
test_iter = 10000

def bench_moe(quant_mode: str):
    with torch.inference_mode(mode=True):
        hidden_type = 30
        if quant_mode == 'fp32':
            gate_type = 0
            up_type = 0
            down_type = 0
            bytes_per_elem = 4.0
        elif quant_mode == 'fp16':
            gate_type = 1
            up_type = 1
            down_type = 1
            bytes_per_elem = 2.0
        elif quant_mode == 'bf16':
            gate_type = 30
            up_type = 30
            down_type = 30
            bytes_per_elem = 2.0
        elif quant_mode == 'q8_0':
            gate_type = 8
            up_type = 8
            down_type = 8
            bytes_per_elem = 1.0625
        elif quant_mode == 'q6_k':
            gate_type = 14
            up_type = 14
            down_type = 14
            bytes_per_elem = 0.820312
        elif quant_mode == 'q5_k_m':
            gate_type = 13
            up_type = 13
            down_type = 14
            bytes_per_elem = 0.731771
        elif quant_mode == 'q4_k_m':
            gate_type = 12
            up_type = 12
            down_type = 14
            bytes_per_elem = 0.648437
        elif quant_mode == 'q3_k_m':
            gate_type = 11
            up_type = 11
            down_type = 13
            bytes_per_elem = 0.515625
        elif quant_mode == 'q2_k':
            gate_type = 10
            up_type = 10
            down_type = 11
            bytes_per_elem = 0.328125
        elif quant_mode == 'iq3_xs':
            gate_type = 21
            up_type = 21
            down_type = 21
            bytes_per_elem = 0.429688
        elif quant_mode == 'iq2_xxs':
            gate_type = 16
            up_type = 16
            down_type = 16
            bytes_per_elem = 0.257812
        else:
            assert False
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device='cuda').to('cpu').contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device='cuda').to('cpu').contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device='cuda').to('cpu').contiguous()
            config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
            moe = cpuinfer_ext.moe.MOE(config)
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)
        expert_ids = torch.stack([torch.stack([torch.randperm(expert_num, dtype=torch.int64, device='cuda')[:n_routed_experts] for _ in range(qlen)]) for _ in range(layer_num)]).to('cpu').contiguous()
        weights = torch.rand((layer_num, qlen, n_routed_experts), dtype=torch.float32, device='cuda').to('cpu').contiguous()
        input = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device='cuda').to('cpu').contiguous()
        output = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device='cuda').to('cpu').contiguous()
        for i in range(warm_up_iter):
            CPUInfer.submit(moes[i % layer_num].forward(qlen, n_routed_experts, expert_ids[i % layer_num].data_ptr(), weights[i % layer_num].data_ptr(), input[i % layer_num].data_ptr(), output[i % layer_num].data_ptr()))
            CPUInfer.sync()
        start = time.perf_counter()
        for i in range(test_iter):
            CPUInfer.submit(moes[i % layer_num].forward(qlen, n_routed_experts, expert_ids[i % layer_num].data_ptr(), weights[i % layer_num].data_ptr(), input[i % layer_num].data_ptr(), output[i % layer_num].data_ptr()))
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start
        print('Quant mode: ', quant_mode)
        print('Time(s): ', total_time)
        print('Iteration: ', test_iter)
        print('Time(us) per iteration: ', total_time / test_iter * 1000000)
        print('Bandwidth: ', hidden_size * intermediate_size * 3 * n_routed_experts * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')
bench_moe('fp32')
bench_moe('fp16')
bench_moe('bf16')
bench_moe('q8_0')
bench_moe('q6_k')
bench_moe('q5_k_m')
bench_moe('q4_k_m')
bench_moe('q3_k_m')
bench_moe('q2_k')