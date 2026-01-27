import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch
input_size = 16384
output_size = 5120
stride = 16
group_max_len = 1024
layer_num = 10
qlen = 1
CPUInfer = cpuinfer_ext.CPUInfer(64)
warm_up_iter = 1000
test_iter = 10000

def bench_linear(quant_mode: str):
    with torch.inference_mode(mode=True):
        hidden_type = 30
        if quant_mode == 'fp32':
            proj_type = 0
            bytes_per_elem = 4.0
        elif quant_mode == 'fp16':
            proj_type = 1
            bytes_per_elem = 2.0
        elif quant_mode == 'bf16':
            proj_type = 30
            bytes_per_elem = 2.0
        elif quant_mode == 'q8_0':
            proj_type = 8
            bytes_per_elem = 1.0625
        elif quant_mode == 'q6_k':
            proj_type = 14
            bytes_per_elem = 0.820312
        elif quant_mode == 'q5_k_m':
            proj_type = 13
            bytes_per_elem = 0.6875
        elif quant_mode == 'q4_k_m':
            proj_type = 12
            bytes_per_elem = 0.5625
        elif quant_mode == 'q3_k_m':
            proj_type = 11
            bytes_per_elem = 0.429688
        elif quant_mode == 'q2_k':
            proj_type = 10
            bytes_per_elem = 0.328125
        elif quant_mode == 'iq3_xs':
            proj_type = 21
            bytes_per_elem = 0.429688
        elif quant_mode == 'iq2_xxs':
            proj_type = 16
            bytes_per_elem = 0.257812
        else:
            assert False
        linears = []
        projs = []
        for _ in range(layer_num):
            proj = torch.randn((output_size, input_size), dtype=torch.float32, device='cuda').to('cpu').contiguous()
            config = cpuinfer_ext.linear.LinearConfig(input_size, output_size, stride, group_max_len, proj.data_ptr(), proj_type, hidden_type)
            linear = cpuinfer_ext.linear.Linear(config)
            projs.append(proj)
            linears.append(linear)
        input = torch.randn((layer_num, qlen, input_size), dtype=torch.bfloat16, device='cuda').to('cpu').contiguous()
        output = torch.empty((layer_num, qlen, output_size), dtype=torch.bfloat16, device='cuda').to('cpu').contiguous()
        for i in range(warm_up_iter):
            CPUInfer.submit(linears[i % layer_num].forward(qlen, input[i % layer_num].data_ptr(), output[i % layer_num].data_ptr()))
            CPUInfer.sync()
        start = time.perf_counter()
        for i in range(test_iter):
            CPUInfer.submit(linears[i % layer_num].forward(qlen, input[i % layer_num].data_ptr(), output[i % layer_num].data_ptr()))
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start
        print('Quant mode: ', quant_mode)
        print('Time(s): ', total_time)
        print('Iteration: ', test_iter)
        print('Time(us) per iteration: ', total_time / test_iter * 1000000)
        print('Bandwidth: ', input_size * output_size * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')
bench_linear('fp32')
bench_linear('fp16')
bench_linear('bf16')
bench_linear('q8_0')
bench_linear('q6_k')
bench_linear('q5_k_m')
bench_linear('q4_k_m')
bench_linear('q3_k_m')
bench_linear('q2_k')