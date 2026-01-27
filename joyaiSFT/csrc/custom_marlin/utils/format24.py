import torch

def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)
    group_x = 64
    group_y = 32 if meta_dtype.itemsize == 2 else 16
    dst_rows = dst_rows // group_x * group_x + dst_rows % 2 * 2 + dst_rows % 8 // 4 + dst_rows % group_y % 4 // 2 * 32 + dst_rows % group_x // 8 * 4
    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)

def sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor')
    m, k = dense.shape
    device = dense.device
    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float, torch.int32]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f'Invalid datatype {dense.dtype} of dense matrix')
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError('Invalid number of elements per meta element calculated')
    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(f'Number of rows of dense matrix {m} must be divisible by 16')
    elif m % 32 != 0:
        raise RuntimeError(f'Number of rows of dense matrix {m} must be divisible by 32')
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(f'Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}')
    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)
    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | bit1.to(torch.int64) << 1
    idxs1 = bit2 | bit3.to(torch.int64) << 1
    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)
    meta_4 = idxs0 | idxs1 << 2
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)
    if quadbits_per_meta_elem == 4:
        meta = meta_n[:, :, 0] | meta_n[:, :, 1] << 4 | meta_n[:, :, 2] << 8 | meta_n[:, :, 3] << 12
    elif quadbits_per_meta_elem == 8:
        meta = meta_n[:, :, 0] | meta_n[:, :, 1] << 4 | meta_n[:, :, 2] << 8 | meta_n[:, :, 3] << 12 | meta_n[:, :, 4] << 16 | meta_n[:, :, 5] << 20 | meta_n[:, :, 6] << 24 | meta_n[:, :, 7] << 28
    meta_reordered = meta.new_empty((m * meta_ncols,))
    meta_offsets = _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device)
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))
    return (sparse, meta_reordered.view(m, meta_ncols))

def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor')
    m, k = sparse.shape
    device = sparse.device
    if meta_reordered.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor')
    if meta_reordered.device != device:
        raise RuntimeError(f'Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device')
    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f'Invalid datatype {meta_dtype} of meta matrix')
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    ksparse = 4 if sparse.dtype != torch.float else 2
    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(f'Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}')
    if meta_ncols * ksparse * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(f'Number of columns of sparse matrix {k} different from the {meta_ncols * ksparse * quadbits_per_meta_elem // 2}, expected according to the number of columns of meta matrix')
    meta_offsets = _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device)
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets).view(m, meta_ncols)
    meta_2 = torch.empty((m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device)
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 3
        meta_2[:, :, 1] = meta >> 2 & 3
        meta_2[:, :, 2] = meta >> 4 & 3
        meta_2[:, :, 3] = meta >> 6 & 3
        meta_2[:, :, 4] = meta >> 8 & 3
        meta_2[:, :, 5] = meta >> 10 & 3
        meta_2[:, :, 6] = meta >> 12 & 3
        meta_2[:, :, 7] = meta >> 14 & 3
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 3
        meta_2[:, :, 1] = meta >> 2 & 3
        meta_2[:, :, 2] = meta >> 4 & 3
        meta_2[:, :, 3] = meta >> 6 & 3
        meta_2[:, :, 4] = meta >> 8 & 3
        meta_2[:, :, 5] = meta >> 10 & 3
        meta_2[:, :, 6] = meta >> 12 & 3
        meta_2[:, :, 7] = meta >> 14 & 3
        meta_2[:, :, 8] = meta >> 16 & 3
        meta_2[:, :, 9] = meta >> 18 & 3
        meta_2[:, :, 10] = meta >> 20 & 3
        meta_2[:, :, 11] = meta >> 22 & 3
        meta_2[:, :, 12] = meta >> 24 & 3
        meta_2[:, :, 13] = meta >> 26 & 3
        meta_2[:, :, 14] = meta >> 28 & 3
        meta_2[:, :, 15] = meta >> 30 & 3
    dense_offsets = meta_2.view(-1) + (torch.arange(0, 2 * m * k // ksparse, device=device) * 4).view(-1, 1).repeat(1, 2).view(-1)
    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    if sparse.dtype != torch.float:
        dense.scatter_(0, dense_offsets, sparse.reshape(-1))
    else:
        dense.view(torch.half).scatter_(0, dense_offsets, sparse.view(torch.half).view(-1))
    return dense.view(m, 2 * k)

def mask_creator(tensor):
    """
    Class for creating N:M sparsity masks.
    Masks will be created using the N:M ratio, where for every block of 
    M weights, N will be pruned based on ranked weight value. Each mask 
    will correspond to the given tensor.

    :param N: The number of weights in a group to keep
    :param M: The size of a weight group
    """
    N = 2
    M = 4
    mask = None
    if tensor.numel() % M != 0:
        raise ValueError(f"Tensor of size {tensor.shape} can't be evenly divided into {M} groups")
    num_groups = tensor.numel() // M
    tensor_temp = tensor.detach().abs().reshape(num_groups, M)
    index = torch.argsort(tensor_temp, dim=1)[:, :int(M - N)]
    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
    mask = w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)
    return mask