import torch
from torch import nn
import warnings
import torch.nn.functional as F
from joyaiSFT.models.configuration_chatrhinoV0 import ChatrhinoV0Config
from joyaiSFT.models.modeling_chatrhinoV0 import ChatrhinoV0Attention, apply_rotary_pos_emb
from typing import Optional, Tuple
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.util.custom_loader import GGUFLoader
from joyaiSFT.util.utils import get_compute_capability
import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from joyaiSFT.util.vendors import device_manager, get_device, to_device, GPUVendor
try:
    from flash_attn import flash_attn_func
except:
    pass
from joyaiSFT.operators.triton_attention import decode_attention_fwd_grouped
from joyaiSFT.operators.triton_attention_prefill import context_attention_fwd
import os
from joyaiSFT.operators.flashinfer_wrapper import flashinfer_enabled
if flashinfer_enabled:
    from joyaiSFT.operators.flashinfer_wrapper import MLAWrapperSingleton
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
from joyaiSFT.models.custom_cache import JOYChatrhinoCache
logger = logging.getLogger('attention')

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class JOYChatrhinoV0Attention(BaseInjectedModule, ChatrhinoV0Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    attn_mask: Optional[torch.Tensor] = None

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', generate_device: str='cuda', chunck_size: int=1000, absorb_for_prefill: bool=False, **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.orig_module.__init__(orig_module.config, orig_module.layer_idx)
        self.chunck_size = chunck_size
        self.mla_wrapper = None
        self.absorb_for_prefill = absorb_for_prefill

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            self.q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
            self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return (self.q_absorb, self.out_absorb)

    def forward_chunck(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since transformer version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            k_pe = k_pe.transpose(1, 2)
            compressed_kv = compressed_kv.unsqueeze(2)
            compressed_kv_with_k_pe, _ = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
            compressed_kv, k_pe = torch.split(compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        q_absorb, out_absorb = self.get_absorbed()
        k_pe = k_pe.view(bsz, 1, -1, self.qk_rope_head_dim)[:, :, :attention_mask.size(-1), :]
        compressed_kv = compressed_kv.view(bsz, 1, -1, self.kv_lora_rank)[:, :, :attention_mask.size(-1), :]
        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.mT)) * self.softmax_scale
        compressed_kv = compressed_kv.squeeze(1)
        '\n        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):\n            raise ValueError(\n                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"\n                f" {attn_weights.size()}"\n            )\n        assert attention_mask is not None\n        '
        if attention_mask is not None:
            '\n            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):\n                raise ValueError(\n                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"\n                )\n            '
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
        attn_output = torch.matmul(attn_output, out_absorb.mT)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        return (attn_output, None, past_key_value)

    def forward_linux_triton(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        compressed_kv = compressed_kv.view(bsz, q_len, 1, self.kv_lora_rank)
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since transformer version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=2)
        if q_len == 1:
            if past_key_value is not None:
                cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
                compressed_kv_with_k_pe, page_table = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
                compressed_kv = compressed_kv_with_k_pe[:, :, :, :self.kv_lora_rank]
            q_absorb, out_absorb = self.get_absorbed()
            q_nope = q_nope.transpose(1, 2)
            q_nope = torch.matmul(q_nope, q_absorb)
            q_nope = q_nope.transpose(1, 2)
            query_states = torch.cat([q_nope, q_pe], dim=-1)
            query_states = query_states.squeeze(1)
            attn_output = torch.zeros_like(q_nope)
            attn_logits = torch.empty((bsz, self.num_heads, 4, self.kv_lora_rank + 1), dtype=torch.float32, device=attn_output.device)
            '\n            print("query_states", torch.isnan(query_states).any())\n            print("compressed_kv_with_k_pe", torch.isnan(compressed_kv_with_k_pe[:,:,0,:]).any())\n            print("compressed_kv", torch.isnan(compressed_kv[:,:,0,:]).any())\n            print("position_ids", torch.isnan(position_ids).any())\n            '
            decode_attention_fwd_grouped(query_states, compressed_kv_with_k_pe, compressed_kv, attn_output, page_table, position_ids.squeeze(0).to(torch.int32) + 1, attn_logits, 4, self.softmax_scale, past_key_value.page_size)
            attn_output = attn_output.transpose(1, 2)
            attn_output = torch.matmul(attn_output, out_absorb.mT)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output)
            return (attn_output, None, past_key_value)
        else:
            if past_key_value is not None:
                cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
                k_pe.squeeze(0)
                compressed_kv.squeeze(0)
                compressed_kv_with_k_pe, _ = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
                compressed_kv, k_pe = torch.split(compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            k_pe = k_pe.view(bsz, -1, self.qk_rope_head_dim)
            k_pe = k_pe[:, :kv_seq_len]
            compressed_kv = compressed_kv.view(bsz, -1, self.kv_lora_rank)
            compressed_kv = compressed_kv[:, :kv_seq_len]
            kv = self.kv_b_proj(compressed_kv).view(bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            query_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
            query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim:] = q_pe
            key_states = k_pe.new_empty(bsz, kv_seq_len, self.num_heads, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe.view(bsz, kv_seq_len, 1, -1)
            value_states = value_states.view(bsz, kv_seq_len, self.num_heads, self.v_head_dim)
            value_states_padded = torch.nn.functional.pad(value_states, [0, query_states.shape[-1] - value_states.shape[-1]], value=0)
            attn_output = flash_attn_func(query_states, key_states, value_states_padded, softmax_scale=self.softmax_scale, causal=True)
            if self.q_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, :self.v_head_dim]
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
            attn_output = self.o_proj(attn_output)
            return (attn_output, None, past_key_value)

    def forward_linux_flashinfer(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.Tensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        compressed_kv = compressed_kv.view(bsz, q_len, 1, self.kv_lora_rank)
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since version transformer verision v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=2)
        if q_len == 1 or self.absorb_for_prefill:
            if past_key_value is not None:
                cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
                compressed_kv_with_k_pe, page_table = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
                compressed_kv = compressed_kv_with_k_pe[:, :, :, :self.kv_lora_rank].view(-1, past_key_value.page_size, self.kv_lora_rank)
                k_pe = compressed_kv_with_k_pe[:, :, :, self.kv_lora_rank:].view(-1, past_key_value.page_size, self.qk_rope_head_dim)
            q_absorb, out_absorb = self.get_absorbed()
            q_nope = q_nope.transpose(1, 2)
            q_nope = torch.matmul(q_nope, q_absorb)
            q_nope = q_nope.transpose(1, 2)
            q_nope = q_nope.contiguous()
            q_nope.squeeze_(0)
            q_pe.squeeze_(0)
            if self.mla_wrapper is None:
                self.mla_wrapper = MLAWrapperSingleton.get_instance(self.device, 1, past_key_value.max_pages, use_cuda_graph=True)
            if self.mla_wrapper.need_plan:
                self.mla_wrapper.need_plan = False
                if q_len == 1:
                    self.mla_wrapper.plan(None, None, None, position_ids.squeeze(1) + 1, None, self.num_heads, self.kv_lora_rank, self.qk_rope_head_dim, past_key_value.page_size, self.softmax_scale, q_nope.dtype, compressed_kv.dtype)
                else:
                    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=self.device)
                    kv_len_arr = torch.tensor([position_ids[0, -1].item() + 1], dtype=torch.int32, device=self.device)
                    self.mla_wrapper.plan(qo_indptr, None, None, kv_len_arr, None, self.num_heads, self.kv_lora_rank, self.qk_rope_head_dim, past_key_value.page_size, self.softmax_scale, q_nope.dtype, compressed_kv.dtype)
            attn_output = self.mla_wrapper.run(q_nope, q_pe, compressed_kv, k_pe).view(bsz, q_len, self.num_heads, self.kv_lora_rank)
            '\n            k = (\n                torch.cat([compressed_kv, k_pe], dim=-1)\n                .view(-1, 1, 512 + 64)\n                .repeat_interleave(self.num_heads, dim=1)\n            )\n            v = compressed_kv.view(-1, 1, 512).repeat_interleave(self.num_heads, dim=1)\n            lens = position_ids.item() + 1\n            #print("lens", lens)\n            attn_ref, lse_ref = attention_ref(\n                1,\n                torch.cat([q_nope, q_pe], dim=-1),\n                k[:lens],\n                v[:lens],\n                False,\n                self.softmax_scale\n            )\n            attn_output = attn_ref.view(bsz, q_len, self.num_heads, self.kv_lora_rank)\n            '
            attn_output = attn_output.transpose(1, 2)
            attn_output = torch.matmul(attn_output, out_absorb.mT)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output)
            return (attn_output, None, past_key_value)
        else:
            if past_key_value is not None:
                cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
                k_pe.squeeze(0)
                compressed_kv.squeeze(0)
                compressed_kv_with_k_pe, _ = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
                compressed_kv, k_pe = torch.split(compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            k_pe = k_pe.view(bsz, -1, self.qk_rope_head_dim)
            k_pe = k_pe[:, :kv_seq_len]
            compressed_kv = compressed_kv.view(bsz, -1, self.kv_lora_rank)
            compressed_kv = compressed_kv[:, :kv_seq_len]
            kv = self.kv_b_proj(compressed_kv).view(bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            query_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
            query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim:] = q_pe
            key_states = k_pe.new_empty(bsz, kv_seq_len, self.num_heads, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe.view(bsz, kv_seq_len, 1, -1)
            value_states = value_states.view(bsz, kv_seq_len, self.num_heads, self.v_head_dim)
            value_states_padded = torch.nn.functional.pad(value_states, [0, query_states.shape[-1] - value_states.shape[-1]], value=0)
            attn_output = flash_attn_func(query_states, key_states, value_states_padded, softmax_scale=self.softmax_scale, causal=True)
            if self.q_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, :self.v_head_dim]
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
            attn_output = self.o_proj(attn_output)
            return (attn_output, None, past_key_value)

    def forward_windows(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
        bsz, q_len, _ = hidden_states.size()
        if q_len <= self.chunck_size:
            return self.forward_chunck(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, **kwargs)
        assert output_attentions == False, 'output_attentions is not supported when using chunked attention'
        attn_output = None
        cur_idx = 0
        while cur_idx < q_len:
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, cur_idx:min(cur_idx + self.chunck_size, q_len), ...]
            else:
                self.attn_mask = torch.zeros(1, 1, self.chunck_size, past_key_value.max_cache_len, device=hidden_states.device) if self.attn_mask is None else self.attn_mask
                self.attn_mask[:, :, :, cur_idx:min(cur_idx + self.chunck_size, past_key_value.max_cache_len)] = -1e+38 * torch.triu(torch.ones(self.chunck_size, self.chunck_size, device=hidden_states.device), diagonal=1)[:, :min(self.chunck_size, min(past_key_value.max_cache_len - cur_idx, self.chunck_size))]
                self.attn_mask[:, :, :, cur_idx + self.chunck_size:] = -1e+38
                self.attn_mask[:, :, :, :cur_idx] = 0
                chunk_mask = torch.narrow(self.attn_mask, 2, 0, min(self.chunck_size, q_len - cur_idx))
            cur_output, _, _ = self.forward_chunck(hidden_states[:, cur_idx:min(cur_idx + self.chunck_size, q_len), ...], chunk_mask, position_ids[:, cur_idx:min(cur_idx + self.chunck_size, q_len)], past_key_value, output_attentions, use_cache, cache_position[cur_idx:min(cur_idx + self.chunck_size, q_len)], **kwargs)
            cur_idx += self.chunck_size
            if attn_output is None:
                attn_output = cur_output
            else:
                attn_output = torch.cat((attn_output, cur_output), dim=-2)
        return (attn_output, None, past_key_value)

    def forward_xpu(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        position_embeddings = kwargs.get('position_embeddings', None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            key_states = torch.cat([k_nope, k_pe.expand([-1, self.num_heads, -1, -1])], dim=-1)
            from ipex_llm.transformers.models.common import rotary_two_with_cache_inplaced
            rotary_two_with_cache_inplaced(query_states[:, :, :, self.qk_nope_head_dim:], key_states[:, :, :, self.qk_nope_head_dim:], cos, sin, True)
        else:
            q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            cos, sin = self.rotary_emb(q_pe, position_ids)
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
            query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim:] = q_pe
            key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}
            key_states, value_states = past_key_value.update(key_states.half(), value_states.half(), self.layer_idx, cache_kwargs)
        attn_weights = None
        from ipex_llm.transformers.models.common import scaled_dot_product_attention
        attn_output = scaled_dot_product_attention(query_states.half(), key_states, value_states, attention_mask.half(), q_len == kv_seq_len, self.softmax_scale)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output).to(hidden_states.dtype)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if torch.xpu.is_available():
            return self.forward_xpu(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, **kwargs)
        elif os.name == 'nt' or get_compute_capability() < 8 or hidden_states.device.type == 'cpu' or (device_manager.gpu_vendor != GPUVendor.NVIDIA):
            return self.forward_windows(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, **kwargs)
        elif flashinfer_enabled:
            return self.forward_linux_flashinfer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, **kwargs)
        else:
            return self.forward_linux_triton(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, **kwargs)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Optional[torch.Tensor], scaling: float, dropout: float=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, attn_weights)