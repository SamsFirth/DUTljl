import torch
from torch import nn
from joyaiSFT.models.modeling_chatrhinoV0 import ChatrhinoV0Attention, apply_rotary_pos_emb
from typing import Optional, Tuple
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.util.custom_loader import GGUFLoader
import logging
from transformers.configuration_utils import PretrainedConfig
from flashinfer import BatchMLAPagedAttentionWrapper
from joyaiSFT.operators.flashinfer_batch_prefill_wrapper import flashInferAttn
from joyaiSFT.models.custom_cache import JOYChatrhinoCache, JOYGQACache
logger = logging.getLogger('attention')

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class flashinfer_attn(BaseInjectedModule, ChatrhinoV0Attention):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', generate_device: str='cuda', chunck_size: int=1000, **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config, orig_module.layer_idx)
        self.chunck_size = chunck_size

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].reshape(-1, self.kv_lora_rank)
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].reshape(-1, self.kv_lora_rank)
            self.q_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False, dtype=q_absorb.dtype, device=q_absorb.device)
            self.q_absorb.weight.data = q_absorb
            self.out_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False, dtype=out_absorb.dtype, device=out_absorb.device)
            self.out_absorb.weight.data = out_absorb
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return (q_absorb, out_absorb)

    def forward(self, hidden_states: torch.Tensor, kv_cache: JOYChatrhinoCache, position_ids: torch.Tensor, wrapper: BatchMLAPagedAttentionWrapper, num_tokens_tensors: torch.Tensor, page_idx: torch.Tensor, page_offset: torch.Tensor):
        q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states, num_tokens_tensors)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states, num_tokens_tensors), num_tokens_tensors), num_tokens_tensors)
        q = q.view(q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states, num_tokens_tensors)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = compressed_kv.contiguous()
        compressed_kv = self.kv_a_layernorm(compressed_kv, num_tokens_tensors)
        k_pe = k_pe.view(q_len, 1, self.qk_rope_head_dim)
        compressed_kv = compressed_kv.view(q_len, 1, self.kv_lora_rank)
        cos, sin = self.rotary_emb(q_pe, position_ids.unsqueeze(0))
        q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q_pe = q_pe.squeeze(0)
        if kv_cache is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'page_idx': page_idx, 'page_offset': page_offset}
            compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, self.layer_idx, page_idx, page_offset, cache_kwargs)
            compressed_kv = compressed_kv_with_k_pe[:, :, :, :self.kv_lora_rank].view(-1, kv_cache.page_size, self.kv_lora_rank)
            k_pe = compressed_kv_with_k_pe[:, :, :, self.kv_lora_rank:].view(-1, kv_cache.page_size, self.qk_rope_head_dim)
        q_absorb, out_absorb = self.get_absorbed()
        q_nope = q_nope.transpose(0, 1)
        q_nope = torch.matmul(q_nope, q_absorb)
        q_nope = q_nope.transpose(0, 1)
        attn_output = wrapper.run(q_nope, q_pe, compressed_kv, k_pe).view(q_len, self.num_heads, self.kv_lora_rank)
        attn_output = attn_output.transpose(0, 1)
        attn_output = torch.matmul(attn_output, out_absorb.mT)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output, num_tokens_tensors)
        return attn_output

class chatrhino_torch_attn(BaseInjectedModule, ChatrhinoV0Attention):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', generate_device: str='cuda', chunck_size: int=1000, **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config, orig_module.layer_idx)
        self.chunck_size = chunck_size

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].reshape(-1, self.kv_lora_rank)
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].reshape(-1, self.kv_lora_rank)
            self.q_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False, dtype=q_absorb.dtype, device=q_absorb.device)
            self.q_absorb.weight.data = q_absorb
            self.out_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False, dtype=out_absorb.dtype, device=out_absorb.device)
            self.out_absorb.weight.data = out_absorb
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return (q_absorb, out_absorb)

    def forward(self, hidden_states: torch.Tensor, kv_cache: JOYChatrhinoCache, position_ids: torch.Tensor, wrapper: None, num_tokens_tensors: torch.Tensor, page_idx: torch.Tensor, page_offset: torch.Tensor, attention_masks: Optional[list[torch.Tensor]]=None, q_indptr: Optional[torch.Tensor]=None, kv_indices: Optional[torch.Tensor]=None, kv_indptr: Optional[torch.Tensor]=None, bsz_tensors: Optional[torch.Tensor]=None, last_page_len: Optional[torch.Tensor]=None):
        final_attention_output = torch.tensor([], device=hidden_states.device)
        for i in range(bsz_tensors[0]):
            batch_num_tokens_tensors = q_indptr[i + 1] - q_indptr[i]
            batch_last_page_len = last_page_len[i]
            batch_page_idx = page_idx[q_indptr[i]:q_indptr[i + 1]]
            batch_page_offset = page_offset[q_indptr[i]:q_indptr[i + 1]]
            kv_page_nums = kv_indptr[i + 1] - kv_indptr[i]
            kv_total_len = kv_page_nums * kv_cache.page_size
            if batch_last_page_len is not None:
                kv_total_len = kv_total_len - (kv_cache.page_size - batch_last_page_len)
            kv_index = kv_indices[kv_indptr[i]:kv_indptr[i + 1]]
            batch_hidden_states = hidden_states[q_indptr[i]:q_indptr[i + 1]]
            batch_position_ids = position_ids[q_indptr[i]:q_indptr[i + 1]]
            q_len, _ = batch_hidden_states.size()
            if self.q_lora_rank is None:
                q = self.q_proj(batch_hidden_states, batch_num_tokens_tensors)
            else:
                q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(batch_hidden_states, batch_num_tokens_tensors), batch_num_tokens_tensors), batch_num_tokens_tensors)
            q = q.view(q_len, self.num_heads, self.q_head_dim)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            compressed_kv = self.kv_a_proj_with_mqa(batch_hidden_states, batch_num_tokens_tensors)
            compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            compressed_kv = compressed_kv.contiguous()
            compressed_kv = self.kv_a_layernorm(compressed_kv, batch_num_tokens_tensors)
            k_pe = k_pe.view(q_len, 1, self.qk_rope_head_dim)
            compressed_kv = compressed_kv.view(q_len, 1, self.kv_lora_rank)
            cos, sin = self.rotary_emb(q_pe, batch_position_ids.unsqueeze(0))
            q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=2)
            q_pe = q_pe.squeeze(0)
            q_pe.transpose_(0, 1)
            if kv_cache is not None:
                cache_kwargs = {'sin': sin, 'cos': cos, 'page_idx': batch_page_idx, 'page_offset': batch_page_offset}
                compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, self.layer_idx, batch_page_idx, batch_page_offset, cache_kwargs)
                compressed_kv = compressed_kv_with_k_pe[:, :, :, :self.kv_lora_rank].view(-1, kv_cache.page_size, self.kv_lora_rank)
                k_pe = compressed_kv_with_k_pe[:, :, :, self.kv_lora_rank:].view(-1, kv_cache.page_size, self.qk_rope_head_dim)
            q_absorb, out_absorb = self.get_absorbed()
            q_nope = q_nope.transpose(0, 1)
            q_nope = torch.matmul(q_nope, q_absorb)
            batch_compressed_kv = None
            batch_k_pe = None
            for page_index in kv_index:
                if kv_total_len > kv_cache.page_size:
                    tmp_compressed_kv = compressed_kv[page_index, 0:kv_cache.page_size, :]
                    tmp_k_pe = k_pe[page_index, 0:kv_cache.page_size, :]
                    if batch_compressed_kv is None or batch_k_pe is None:
                        batch_compressed_kv = tmp_compressed_kv
                        batch_k_pe = tmp_k_pe
                    else:
                        batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
                        batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
                    kv_total_len -= kv_cache.page_size
                else:
                    tmp_compressed_kv = compressed_kv[page_index, 0:kv_total_len, :]
                    tmp_k_pe = k_pe[page_index, 0:kv_total_len, :]
                    if batch_compressed_kv is None or batch_k_pe is None:
                        batch_compressed_kv = tmp_compressed_kv
                        batch_k_pe = tmp_k_pe
                    else:
                        batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
                        batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
                    break
            attention_weights = (torch.matmul(q_pe, batch_k_pe.mT) + torch.matmul(q_nope, batch_compressed_kv.mT)) * self.softmax_scale
            attention_weights = attention_weights + attention_masks[i]
            attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
            attn_output = torch.matmul(attention_weights, batch_compressed_kv)
            out_absorb = out_absorb.transpose(1, 2)
            attn_output = torch.matmul(attn_output, out_absorb)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output, batch_num_tokens_tensors)
            final_attention_output = torch.cat((final_attention_output, attn_output), dim=0)
        return final_attention_output