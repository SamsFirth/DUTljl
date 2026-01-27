import argparse
import torch
import sys
import os
from piped_subprocess import PipedSubprocess, TORCH_DTYPE_NAME
import math
parser = argparse.ArgumentParser()
parser.add_argument('example_exe', type=str, help='Path to the 41_fused_multi_head_attention_backward executable')
args = parser.parse_args()
torch.manual_seed(0)
dtype = torch.float16
B, Mq, Mkv, H, K, Kv = (2, 1024, 1024, 5, 128, 128)
causal = True
repeat_count = 100
ATOL = {torch.float: 0.0005, torch.half: 0.095, torch.bfloat16: 0.7}[dtype]
RTOL = {torch.float: 0.0001, torch.half: 0.02, torch.bfloat16: 0.1}[dtype]
assert not (causal and Mq < Mkv), 'causal only supports seqlenK <= seqlenQ'
fmha_bw_binary = args.example_exe
if not os.path.isfile(fmha_bw_binary):
    print(f'No such file: `{fmha_bw_binary}`\nDid you forget to run "make 41_fused_multi_head_attention"?')
    sys.exit(1)

def create_lower_triangular_mask():
    return torch.triu(torch.full([1, Mq, Mkv], dtype=dtype, fill_value=float('-inf')), diagonal=1)

def ref_mha_bmk(q, k, v, mask):
    q = q.float()
    k = k.float()
    v = v.float()
    q = q * (1 / q.shape[-1] ** 0.5)
    attn = q @ k.transpose(-2, -1)
    if mask is not None:
        attn += mask
    attn_max = attn.max(-1, True).values
    attn_norm = (attn - attn_max).exp().sum(-1, True)
    attn = attn.softmax(-1)
    lse = attn_max + attn_norm.log()
    lse = lse.squeeze(2)
    return (attn @ v, lse)

def bmhk2bmk(t):
    return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

def ref_mha_bmhk(q, k, v, mask):
    assert q.ndim == 4
    out, lse = ref_mha_bmk(bmhk2bmk(q), bmhk2bmk(k), bmhk2bmk(v), mask=mask)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return (out.permute((0, 2, 1, 3)), lse.reshape([q.shape[0], q.shape[2], q.shape[1]]))

def ref_mha_bw_bmhk(q, k, v, mask, lse, out, grad_out, delta):
    lse = lse[:, :, :q.shape[1]]
    delta = delta.reshape([-1, delta.shape[-1], 1])
    q, k, v, out, grad_out = [bmhk2bmk(x).float() for x in (q, k, v, out, grad_out)]
    attn_T = k @ q.transpose(-2, -1)
    if mask is not None:
        attn_T += mask.transpose(-2, -1)
    attn_T = attn_T * (1 / q.shape[-1] ** 0.5)
    attn_T = attn_T - lse.reshape([-1, 1, lse.shape[-1]])
    attn_T = attn_T.exp()
    grad_v = attn_T @ grad_out
    dov = grad_out @ v.transpose(-2, -1)
    tmp = (dov - delta) * attn_T.transpose(-2, -1)
    tmp = tmp / q.shape[-1] ** 0.5
    grad_q = tmp @ k
    grad_k = tmp.transpose(-2, -1) @ q
    return [x.reshape([B, H, x.shape[1], x.shape[-1]]).permute([0, 2, 1, 3]) for x in [grad_q, grad_k, grad_v]]
print('initializing tensors...')
query = torch.randn([B, Mq, H, K], dtype=dtype)
key = 3 * torch.randn([B, Mkv, H, K], dtype=dtype)
value = 3 * torch.randn([B, Mkv, H, Kv], dtype=dtype)
mask = create_lower_triangular_mask() if causal else None
query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)
print('computing fw...')
out, lse = ref_mha_bmhk(query, key, value, mask=mask)
out = out.to(dtype).contiguous()
grad_out = 3 * torch.randn([B, Mq, H, Kv], dtype=dtype)
print('computing bw with autograd...')
out.backward(grad_out)
scale = 1 / query.shape[-1] ** 0.5
delta = (grad_out.float() * out.float()).sum(-1).transpose(-2, -1).contiguous()
pad_amount = (32 - lse.shape[2] % 32) % 32
lse = torch.nn.functional.pad(lse, [0, pad_amount], value=math.inf)
print('computing bw with reference implem...')
gQr, gKr, gVr = ref_mha_bw_bmhk(query, key, value, mask, lse, out, grad_out, delta)
with PipedSubprocess(fmha_bw_binary) as bw_kernel:
    bw_kernel.write(TORCH_DTYPE_NAME[query.dtype], 'scale', scale, 'head_dim', K, 'head_dim_value', Kv, 'num_queries', Mq, 'num_keys', Mkv, 'num_heads', H, 'custom_mask_type', 1 if causal else 0, 'num_batches', B, 'repeat_count', repeat_count, 'num_splits_key', Mkv // 128)
    bw_kernel.writeTensor(query, 'query', ['q_strideB', 'q_strideM', 'q_strideH'])
    bw_kernel.writeTensor(key, 'key', ['k_strideB', 'k_strideM', 'k_strideH'])
    bw_kernel.writeTensor(value, 'value', ['v_strideB', 'v_strideM', 'v_strideH'])
    bw_kernel.writeTensor(lse, 'logsumexp', ['lse_strideB', 'lse_strideH'])
    bw_kernel.writeTensor(out, 'output', ['o_strideB', 'o_strideM', 'o_strideH'])
    bw_kernel.writeTensor(grad_out, 'grad_output', ['gO_strideB', 'gO_strideM', 'gO_strideH'])
    bw_kernel.writeTensor(delta, 'delta', ['delta_strideB', 'delta_strideH'])
    if bw_kernel.read() != 'OK':
        print('Got unexpected output')
        print(bw_kernel.subp.communicate()[0])
        sys.exit(0)
    gQ = bw_kernel.readTensor('grad_query', ['gQ_strideB', 'gQ_strideM', 'gQ_strideH'], query.shape).float()
    gK = bw_kernel.readTensor('grad_key', ['gK_strideB', 'gK_strideM', 'gK_strideH'], key.shape).float()
    gV = bw_kernel.readTensor('grad_value', ['gV_strideB', 'gV_strideM', 'gV_strideH'], value.shape).float()
    runtime_ms = float(bw_kernel.readNamed('runtime_ms'))
float_ops = B * H * sum([Mq * Mkv * K * 2, Mkv * Mq * Kv * 2, Mq * Kv * Mkv * 2, Mq * K * Mkv * 2, Mq * K * Mkv * 2])
if causal:
    float_ops //= 2
print(f"\nFused multi-head attention - backward\n    batch_size={B}\n    num_queries={Mq}\n    num_keys={Mkv}\n    num_heads={H}\n    head_dim={K}\n    head_dim_value={Kv}\n\n    Correctness:\n        grad_query: {('PASS' if torch.allclose(gQ, gQr, rtol=RTOL, atol=ATOL) else 'FAIL')} (delta: {(gQ - gQr).abs().max()})\n        grad_key:   {('PASS' if torch.allclose(gK, gKr, rtol=RTOL, atol=ATOL) else 'FAIL')} (delta: {(gK - gKr).abs().max()})\n        grad_value: {('PASS' if torch.allclose(gV, gVr, rtol=RTOL, atol=ATOL) else 'FAIL')} (delta: {(gV - gVr).abs().max()})\n        (atol={ATOL} / rtol={RTOL})\n    Runtime: {runtime_ms}ms ({float_ops / 1024 ** 4 / (runtime_ms / 1000):.4f} TFlops)\n")
assert torch.allclose(query.grad.float(), gQr, rtol=RTOL, atol=ATOL), 'Reference implementation does not match PyTorch autograd!'
assert torch.allclose(key.grad.float(), gKr, rtol=RTOL, atol=ATOL), 'Reference implementation does not match PyTorch autograd!'
assert torch.allclose(value.grad.float(), gVr, rtol=RTOL, atol=ATOL), 'Reference implementation does not match PyTorch autograd!'