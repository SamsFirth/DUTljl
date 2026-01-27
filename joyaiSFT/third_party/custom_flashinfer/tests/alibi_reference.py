"""
Attention with Linear Biases (ALiBi) reference implementation.

Code adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations

Licensed under MIT, you may obtain a copy of the License at

  https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license

Source:
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/285cb3735bde02fbc8c19ddeb24d0ae7e77135c1/labml_nn/transformers/mha.py
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/285cb3735bde02fbc8c19ddeb24d0ae7e77135c1/labml_nn/transformers/alibi/__init__.py
"""
import math
from typing import Optional
import torch

def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\x0crac{1}{2^{\x0crac{8}{n}}} = 2^{-\x0crac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\x0crac{1}{2^1}, \x0crac{1}{2^2}, \\dots, \x0crac{1}{2^8}$$
    """
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m

@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """
    m = get_slopes(n_heads).to(mask.device)
    distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[None, :]
    return distance[:, :, None] * m[None, None, :]

def alibi_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]=None):
    """
    query: [q_len, num_heads, head_dim]
    key: [kv_len, num_heads, head_dim]
    value: [kv_len, num_heads, head_dim]
    mask: [q_len, kv_len]
    """
    q_len, num_heads, head_dim = query.shape
    scores = torch.einsum('qhd,khd->qkh', query.float(), key.float())
    scores *= 1.0 / math.sqrt(head_dim)
    alibi_biases = get_alibi_biases(num_heads, mask)
    scores += alibi_biases
    scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
    attn = torch.softmax(scores, dim=1)
    return torch.einsum('ovh,vhd->ohd', attn, value.float()).to(query)