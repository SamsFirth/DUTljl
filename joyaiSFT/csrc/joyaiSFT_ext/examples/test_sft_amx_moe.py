
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch
from pathlib import Path
import numpy as np
expert_num = 10
hidden_size = 5120
intermediate_size = 1536
max_len = 1024
n_routed_experts = 2
qlen = 600
layer_num = 10
num_threads = 112
validation_iter = 1
LAYER_IDX = 0
DUMP_DIR = Path(os.getenv('SFT_DEBUG_PATH', 'debug'))
dtype = torch.bfloat16
gradtype = torch.bfloat16
import shutil
folder_path = '/home/lpl/kt-sft/debug'
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

def silu_fwd(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))

def silu_grad(x: torch.Tensor) -> torch.Tensor:
    """SiLU激活函数的梯度"""
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))

class SiLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return silu_fwd(inp)

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        sig = torch.sigmoid(inp)
        return grad_out * (sig + inp * sig * (1.0 - sig))
silu = SiLU.apply

def mlp_torch(x, gate, up, down, req_grad=False):
    g = torch.mm(x, gate.t())
    u = torch.mm(x, up.t())
    if req_grad:
        inter = silu_fwd(g) * u
    else:
        inter = silu_fwd(g) * u
    return torch.mm(inter, down.t())

def moe_torch(x, eid, w, gate, up, down, req_grad=False):
    """eid: [T,k]  int64,  w: [T,k] float"""
    T, k = eid.shape
    tok_cnt = torch.zeros(expert_num, dtype=torch.int64)
    for e in eid.view(-1):
        tok_cnt[e] += 1
    order = eid.view(-1).argsort()
    packed = x[order // k]
    outputs, start = ([], 0)
    for e in range(expert_num):
        num = tok_cnt[e].item()
        if not num:
            continue
        end = start + num
        o = mlp_torch(packed[start:end], gate[e], up[e], down[e], req_grad)
        outputs.append(o)
        start = end
    if outputs:
        out_all = torch.cat(outputs, 0)
    else:
        out_all = packed.new_empty(0, hidden_size)
    out_restore = torch.empty_like(out_all)
    out_restore[order] = out_all
    out_restore = out_restore.view(T, k, hidden_size)
    out = (out_restore * w.unsqueeze(-1)).sum(1)
    return out

def moe_backward_python(x, eid, w, gate, up, down, grad_output, gate_u_cache, up_v_cache):
    """
    Python模拟C++的MoE backward计算 - 完全仿照sft_moe.hpp的实现
    参数:
        x: 输入 [T, hidden_size]
        eid: expert_ids [T, k]
        w: weights [T, k]
        gate, up, down: 权重矩阵
        grad_output: 输出梯度 [T, hidden_size]
        gate_u_cache, up_v_cache: forward时缓存的中间结果
    返回:
        grad_input: 输入梯度 [T, hidden_size]
    """
    T, k = eid.shape
    expert_num = gate.shape[0]
    hidden_size = gate.shape[2]
    intermediate_size = gate.shape[1]
    print('\n========== Python Backward详细对拍 ==========')
    print(f'输入形状: T={T}, k={k}, hidden_size={hidden_size}, intermediate_size={intermediate_size}')
    print(f'\n--- Python Token 0 ---')
    print(f'  Expert 0: weight={w[0, 0].item():.6f}')
    grad_input = torch.zeros_like(x, dtype=torch.float32)
    expert_token_counts = torch.zeros(expert_num, dtype=torch.int64)
    for i in range(T):
        for j in range(k):
            expert_token_counts[eid[i, j]] += 1
    expert_token_indices = [[] for _ in range(expert_num)]
    expert_token_positions = [[] for _ in range(expert_num)]
    for i in range(T):
        for j in range(k):
            expert_id = int(eid[i, j].item())
            expert_token_indices[expert_id].append(i)
            expert_token_positions[expert_id].append(j)
    max_tokens_per_expert = int(expert_token_counts.max().item()) if expert_token_counts.max() > 0 else 0
    local_input = torch.zeros(expert_num, max_tokens_per_expert, hidden_size, dtype=torch.float32)
    local_gate_output = torch.zeros(expert_num, max_tokens_per_expert, intermediate_size, dtype=torch.float32)
    local_up_output = torch.zeros(expert_num, max_tokens_per_expert, intermediate_size, dtype=torch.float32)
    local_down_output_grad = torch.zeros(expert_num, max_tokens_per_expert, hidden_size, dtype=torch.float32)
    local_down_input_grad = torch.zeros(expert_num, max_tokens_per_expert, intermediate_size, dtype=torch.float32)
    local_gate_output_grad = torch.zeros(expert_num, max_tokens_per_expert, intermediate_size, dtype=torch.float32)
    local_up_output_grad = torch.zeros(expert_num, max_tokens_per_expert, intermediate_size, dtype=torch.float32)
    local_gate_input_grad = torch.zeros(expert_num, max_tokens_per_expert, hidden_size, dtype=torch.float32)
    local_up_input_grad = torch.zeros(expert_num, max_tokens_per_expert, hidden_size, dtype=torch.float32)
    for expert_id in range(expert_num):
        for local_idx, (token_idx, expert_pos) in enumerate(zip(expert_token_indices[expert_id], expert_token_positions[expert_id])):
            local_input[expert_id, local_idx] = x[token_idx].to(torch.float32)
            local_down_output_grad[expert_id, local_idx] = grad_output[token_idx].to(torch.float32)
    for expert_id in range(expert_num):
        num_tokens = expert_token_counts[expert_id]
        if num_tokens == 0:
            continue
        local_input_expert = local_input[expert_id, :num_tokens]
        gate_output = torch.mm(local_input_expert, gate[expert_id].to(torch.float32).t())
        up_output = torch.mm(local_input_expert, up[expert_id].to(torch.float32).t())
        gate_output_activated = silu_fwd(gate_output) * up_output
        local_gate_output[expert_id, :num_tokens] = gate_output
        local_up_output[expert_id, :num_tokens] = up_output
    for expert_id in range(expert_num):
        num_tokens = expert_token_counts[expert_id]
        if num_tokens == 0:
            continue
    for expert_id in range(expert_num):
        num_tokens = expert_token_counts[expert_id]
        if num_tokens == 0:
            continue
        down_input_grad = torch.mm(local_down_output_grad[expert_id, :num_tokens], down[expert_id].to(torch.float32))
        local_down_input_grad[expert_id, :num_tokens] = down_input_grad
    for expert_id in range(expert_num):
        num_tokens = expert_token_counts[expert_id]
        if num_tokens == 0:
            continue
        torch.save(local_down_output_grad[expert_id, :num_tokens], f'debug/py_layer0_E_End{expert_id}_down_output_grad_.pt')
        torch.save(local_gate_output[expert_id, :num_tokens], f'debug/py_layer0_E_End{expert_id}_gate_output_.pt')
    for expert_id in range(expert_num):
        num_tokens = expert_token_counts[expert_id]
        if num_tokens == 0:
            continue
        for local_idx in range(num_tokens):
            token_idx = expert_token_indices[expert_id][local_idx]
            expert_pos = expert_token_positions[expert_id][local_idx]
            weight = w[token_idx, expert_pos].item()
            should_print = token_idx == 0 and expert_pos == 0
            gate_u = local_gate_output[expert_id, local_idx]
            up_v = local_up_output[expert_id, local_idx]
            down_input_grad_token = local_down_input_grad[expert_id, local_idx]
            down_input_grad_token = down_input_grad_token * weight
            if should_print:
                print(f'    down_input_grad前5个值: {down_input_grad_token[:5].tolist()}')
            gate_output_grad = down_input_grad_token * up_v * silu_grad(gate_u)
            up_output_grad = down_input_grad_token * silu_fwd(gate_u)
            if should_print:
                print(f'    gate_output_grad前5个值: {gate_output_grad[:5].tolist()}')
                print(f'    up_output_grad前5个值: {up_output_grad[:5].tolist()}')
            local_gate_output_grad[expert_id, local_idx] = gate_output_grad
            local_up_output_grad[expert_id, local_idx] = up_output_grad
    for expert_id in range(expert_num):
        num_tokens = expert_token_counts[expert_id]
        if num_tokens == 0:
            continue
        gate_input_grad = torch.mm(local_gate_output_grad[expert_id, :num_tokens], gate[expert_id].to(torch.float32))
        up_input_grad = torch.mm(local_up_output_grad[expert_id, :num_tokens], up[expert_id].to(torch.float32))
        local_gate_input_grad[expert_id, :num_tokens] = gate_input_grad
        local_up_input_grad[expert_id, :num_tokens] = up_input_grad
        if expert_id == 0 and num_tokens > 0:
            token_idx = expert_token_indices[expert_id][0]
            expert_pos = expert_token_positions[expert_id][0]
            if token_idx == 0 and expert_pos == 0:
                print(f'    gate_input_grad前5个值: {gate_input_grad[0, :5].tolist()}')
                print(f'    up_input_grad前5个值: {up_input_grad[0, :5].tolist()}')
    for token_idx in range(T):
        token_grad = torch.zeros(hidden_size, dtype=torch.float32)
        for expert_pos in range(k):
            expert_id = int(eid[token_idx, expert_pos].item())
            local_idx = expert_token_indices[expert_id].index(token_idx)
            token_grad += local_gate_input_grad[expert_id, local_idx]
            token_grad += local_up_input_grad[expert_id, local_idx]
        grad_input[token_idx] = token_grad
        if token_idx == 0:
            print(f'  Token 0 最终input_grad前5个值: {token_grad[:5].tolist()}')
    return grad_input

def test_amx_moe_two_round():
    gate_proj = torch.randn(expert_num, intermediate_size, hidden_size, dtype=torch.bfloat16, requires_grad=True).contiguous()
    up_proj = torch.randn_like(gate_proj)
    down_proj = torch.randn(expert_num, hidden_size, intermediate_size, dtype=torch.bfloat16, requires_grad=True).contiguous()
    cfg = cpuinfer_ext.sft_moe.SFT_AMX_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr())
    moe_cpp = cpuinfer_ext.sft_moe.SFT_AMXInt8_MOE(cfg)
    cpu_infer = cpuinfer_ext.CPUInfer(num_threads)
    cpu_infer.submit(moe_cpp.load_weights())
    cpu_infer.sync()
    expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand(qlen, n_routed_experts, dtype=torch.float32).contiguous()
    input_pt = (torch.randn((qlen, hidden_size), dtype=dtype) / 100).detach().requires_grad_(True).contiguous()
    input_cpp = input_pt.detach().clone().requires_grad_(True).contiguous()
    out_ref = moe_torch(input_pt, expert_ids, weights, gate_proj, up_proj, down_proj, True)
    out_ref.retain_grad()
    gate_u_cache = []
    up_v_cache = []
    for token_idx in range(qlen):
        token_gate_u = []
        token_up_v = []
        for expert_pos in range(n_routed_experts):
            expert_id = int(expert_ids[token_idx, expert_pos].item())
            gate_u = torch.mm(input_pt[token_idx:token_idx + 1].to(torch.float32), gate_proj[expert_id].to(torch.float32).t()).squeeze()
            up_v = torch.mm(input_pt[token_idx:token_idx + 1].to(torch.float32), up_proj[expert_id].to(torch.float32).t()).squeeze()
            token_gate_u.append(gate_u)
            token_up_v.append(up_v)
        gate_u_cache.append(token_gate_u)
        up_v_cache.append(token_up_v)
    flop_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    flop_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    out_cpp = torch.empty_like(out_ref, dtype=dtype).contiguous()
    t0 = time.time()
    cpu_infer.submit(moe_cpp.forward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), out_cpp.data_ptr()))
    cpu_infer.sync()
    t1 = time.time()
    diff_fwd = (out_cpp.to(torch.float32) - out_ref.to(torch.float32)).abs()
    print(f'out_cpp.to(torch.float32):{out_cpp.to(torch.float32)}, out_ref.to(torch.float32):{out_ref.to(torch.float32)}')
    rel_fwd = diff_fwd.mean() / out_ref.abs().mean()
    print(f'Forward   diff: {rel_fwd.item():.3e} | time {t1 - t0:.4f}s | TFLOPS {flop_fwd / (t1 - t0) / 1000000000000.0:.2f}')
    grad_out = torch.randn_like(out_ref, dtype=gradtype).contiguous()
    grad_out_cpp = grad_out.clone().contiguous()
    grad_in_cpp = torch.zeros_like(input_cpp, dtype=gradtype).contiguous()
    for p in (gate_proj, up_proj, down_proj, input_pt):
        if p.grad is not None:
            p.grad.zero_()
    t2 = time.time()
    out_ref.backward(grad_out, retain_graph=True)
    t3 = time.time()
    print(f'PyTorch backward time {t3 - t2:.4f}s | TFLOPS {flop_bwd / (t3 - t2) / 1000000000000.0:.2f}')
    t4_py = time.time()
    grad_in_python = moe_backward_python(input_pt, expert_ids, weights, gate_proj, up_proj, down_proj, grad_out.to(torch.float32), gate_u_cache, up_v_cache)
    t5_py = time.time()
    print(f'Python   backward time {t5_py - t4_py:.4f}s | TFLOPS {flop_bwd / (t5_py - t4_py) / 1000000000000.0:.2f}')
    t4 = time.time()
    print('Before backward')
    cpu_infer.submit(moe_cpp.backward(qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(), grad_out_cpp.data_ptr(), grad_in_cpp.data_ptr()))
    cpu_infer.sync()
    t5 = time.time()
    print('After backward')
    print(f'C++      backward time {t5 - t4:.4f}s | TFLOPS {flop_bwd / (t5 - t4) / 1000000000000.0:.2f}')
    gcpp = grad_in_cpp.to(torch.float32)
    gref = input_pt.grad.to(torch.float32) if input_pt.grad is not None else torch.zeros_like(input_pt, dtype=torch.float32)
    gpy = grad_in_python.to(torch.float32)
    print(f'C++ AMX backward:{gcpp}', '\n', '\n', f'python backward:{gpy}')
    rel_bwd_cpp = (gcpp - gref).abs().mean() / gref.abs().mean()
    rel_bwd_py = (gpy - gref).abs().mean() / gref.abs().mean()
    rel_bwd_cpp_py = (gcpp - gpy).abs().mean() / gpy.abs().mean()
    print(f'Torch vs C++:    {rel_bwd_cpp.item():.3e}')
    print(f'Torch vs Python: {rel_bwd_py.item():.3e}')
    print(f'C++ vs Python:   {rel_bwd_cpp_py.item():.3e}')
    if rel_bwd_cpp_py.item() < 0.05:
        print('✅ C++和Python backward对拍成功!')
    else:
        print('❌ C++和Python backward对拍失败，存在显著差异')

def load_bf16(stub, shape):
    with open(stub + '.bf16', 'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.bfloat16).view(shape).float()

def load_f16(stub, shape):
    with open(stub + '.f16', 'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.float16).view(shape).float()

def load_f32(stub, shape):
    with open(stub + '.f32', 'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.float32).view(shape)

def load_uint8(stub, shape):
    with open(stub + '.uint8', 'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.uint8).view(shape)

def load_int8(stub, shape):
    with open(stub + '.int8', 'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.int8).view(shape)

def load_dump_tensor(experts_idx: int, name: str, shape: tuple, Ename: str='E_Before'):
    """
    根据 experts_idx / name / shape 读取 dump 文件，并返回 torch.Tensor
    """
    stub = DUMP_DIR / f'layer{LAYER_IDX}_{Ename}{experts_idx}_{name}'
    if stub.with_suffix('.bf16').exists():
        return load_bf16(str(stub), shape)
    elif stub.with_suffix('.f16').exists():
        return load_f16(str(stub), shape)
    elif stub.with_suffix('.f32').exists():
        return load_f32(str(stub), shape)
    elif stub.with_suffix('.uint8').exists():
        return load_uint8(str(stub), shape)
    elif stub.with_suffix('.int8').exists():
        return load_int8(str(stub), shape)
    else:
        raise FileNotFoundError(f'{stub}（bf16/f16/f32/u8/i8 均不存在）')

def load_bin(path, n, k):
    data = np.fromfile(path, dtype=np.float32)
    assert data.size == n * k
    data = data.reshape(n, k)
    return torch.from_numpy(data).to(torch.bfloat16)

def check_nan(name, shape):
    stub1 = DUMP_DIR / f'{name}'
    if stub1.with_suffix('.bf16').exists():
        cpp_bef = load_bf16(str(stub1), shape)
    elif stub1.with_suffix('.f16').exists():
        cpp_bef = load_f16(str(stub1), shape)
    elif stub1.with_suffix('.f32').exists():
        cpp_bef = load_f32(str(stub1), shape)
    elif stub1.with_suffix('.int8').exists():
        return load_int8(str(stub1), shape)
    else:
        print('dump 缺失/未知类型')
        return
    print(f'{name}:{cpp_bef}')
    print(f' shape : {cpp_bef.shape}')
    print(f' dtype : {cpp_bef.dtype}')
    finite_mask = torch.isfinite(cpp_bef)
    if finite_mask.any():
        t_finite = cpp_bef[finite_mask]
        t_max = t_finite.max().item()
        t_min = t_finite.min().item()
        print(f' max   : {t_max:.6e}')
        print(f' min   : {t_min:.6e}')
    else:
        print(' max/min: 所有元素均为 NaN / Inf')
    for nan_name, t in [(f'{name}', cpp_bef)]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f'{name} 含 NaN={nan_cnt}、Inf={inf_cnt}')
        else:
            print('NO NaN or Inf exist')

def get_tensor(name, shape) -> torch.Tensor:
    stub1 = DUMP_DIR / f'{name}'
    if stub1.with_suffix('.bf16').exists():
        cpp_bef = load_bf16(str(stub1), shape)
    elif stub1.with_suffix('.f16').exists():
        cpp_bef = load_f16(str(stub1), shape)
    elif stub1.with_suffix('.f32').exists():
        cpp_bef = load_f32(str(stub1), shape)
    elif stub1.with_suffix('.int8').exists():
        return load_int8(str(stub1), shape)
    else:
        print('dump 缺失/未知类型')
        return
    return cpp_bef

def check_py_cpp(name1, name2, shape):
    print(f'compare {name1} with {name2}, at shape{shape}')
    stub1 = DUMP_DIR / f'{name1}'
    py_bef = torch.load(f'{stub1}')
    if not isinstance(py_bef, torch.Tensor):
        print(f'⚠️ {name1} 不是 Tensor，而是 {type(py_bef)}')
        return
    stub2 = DUMP_DIR / f'{name2}'
    if stub2.with_suffix('.bf16').exists():
        cpp_bef = load_bf16(str(stub2), shape)
    elif stub2.with_suffix('.f16').exists():
        cpp_bef = load_f16(str(stub2), shape)
    elif stub2.with_suffix('.f32').exists():
        cpp_bef = load_f32(str(stub2), shape)
    elif stub2.with_suffix('.int8').exists():
        return load_int8(str(stub2), shape)
    else:
        print(f'dump 缺失/未知类型: {stub2}')
        return
    for t in [py_bef]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f'{name1} 含 NaN={nan_cnt}、Inf={inf_cnt}')
        else:
            print('NO NaN or Inf exist')
    for t in [cpp_bef]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f'{name2} 含 NaN={nan_cnt}、Inf={inf_cnt}')
        else:
            print('NO NaN or Inf exist')
    if py_bef.shape != cpp_bef.shape:
        print(f'shape 不一致: py_bef {py_bef.shape}, cpp_bef {cpp_bef.shape}')
    else:
        eps = 1e-06
        denominator = torch.abs(py_bef) + eps
        rel_diff = torch.abs(py_bef - cpp_bef) / denominator
        mask = rel_diff > 0.02
        num_large_diff = mask.sum().item()
        total = rel_diff.numel()
        if num_large_diff == 0:
            print('✅ 所有元素相对误差都在 2% 范围内')
            flat_rel_diff = rel_diff.view(-1)
            max_idx = torch.argmax(flat_rel_diff)
            max_val = flat_rel_diff[max_idx].item()
            max_pos = tuple(torch.unravel_index(max_idx, py_bef.shape))
            py_val = py_bef[max_pos].item()
            cpp_val = cpp_bef[max_pos].item()
            print(f'    最大相对误差 = {max_val:.2%}')
            print(f'    最大相对误差位置: {max_pos}, py  = {py_val:.6f}, cpp = {cpp_val:.6f}')
        else:
            print(f'❗ 相对误差 > 2% 的元素数量: {num_large_diff} / {total}')
            print(f'{name1}: {py_bef}')
            print(f'{name2}: {cpp_bef}')

def manual_check(experts_ids):
    expert_token_counts = torch.zeros(expert_num, dtype=torch.int64)
    T, k = experts_ids.shape
    for i in range(T):
        for j in range(k):
            expert_token_counts[experts_ids[i, j]] += 1
    for experts_idx in range(expert_num):
        down_ba_ori = get_tensor(f'cpp_layer0_E_End{experts_idx}_down_ba_ori_', (expert_token_counts[experts_idx], intermediate_size))
        down_output_grad = get_tensor(f'cpp_layer0_E_End{experts_idx}_down_output_grad_', (expert_token_counts[experts_idx], hidden_size))
        py_down_t_ba = torch.load(f'debug/py_layer0_E_End{experts_idx}_down_output_grad_.pt')
        py_down_ba = torch.load(f'debug/py_layer0_E_End{experts_idx}_gate_output_.pt')
        print(f'cpp_{experts_idx}_down_ba_ori_:{down_ba_ori}')
        print(f'py_{experts_idx}_down_ba_ori_view: {py_down_ba}')
        print(f'cpp_{experts_idx}_down_t_ba_ori_view:{down_output_grad}')
        print(f'py_{experts_idx}_down_t_ba_ori_view: {py_down_t_ba}')
if __name__ == '__main__':
    torch.manual_seed(42)
    test_amx_moe_two_round()