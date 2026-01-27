from torch import nn
from transformers import ROPE_INIT_FUNCTIONS
from joyaiSFT.models.modeling_chatrhino import ChatrhinoRotaryEmbedding
from joyaiSFT.models.modeling_chatrhinoV0 import ChatrhinoV0YarnRotaryEmbedding, ChatrhinoV0RotaryEmbedding, yarn_get_mscale, yarn_linear_ramp_mask, yarn_find_correction_range
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.util.custom_loader import GGUFLoader
from joyaiSFT.util.inference_state import InferenceState
from joyaiSFT.util.grad_wrapper import maybe_no_grad
from transformers.configuration_utils import PretrainedConfig
import torch

class RotaryEmbedding(BaseInjectedModule, ChatrhinoV0RotaryEmbedding):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, generate_device: str='cuda', prefill_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.orig_module.__init__(orig_module.dim, orig_module.max_position_embeddings, orig_module.base)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        self.orig_module.__init__(self.orig_module.dim, self.orig_module.max_position_embeddings, self.orig_module.base, self.device)

class RotaryEmbeddingV3(BaseInjectedModule):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, generate_device: str='cuda', prefill_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    @maybe_no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

    def load(self):
        self._init(dim=self.config.qk_rope_head_dim, max_position_embeddings=self.config.max_position_embeddings, base=self.config.rope_theta, device=self.device)

    def _init(self, dim, max_position_embeddings, base, device, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        self.max_seq_len_cached = max_position_embeddings

class YarnRotaryEmbedding(BaseInjectedModule, ChatrhinoV0YarnRotaryEmbedding):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, generate_device: str='cuda', prefill_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.orig_module.__init__(orig_module.dim, orig_module.max_position_embeddings, orig_module.base, None, orig_module.scaling_factor, orig_module.original_max_position_embeddings, orig_module.beta_fast, orig_module.beta_slow, orig_module.mscale, orig_module.mscale_all_dim)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        self.orig_module.__init__(self.orig_module.dim, self.orig_module.max_position_embeddings, self.orig_module.base, self.generate_device, self.orig_module.scaling_factor, self.orig_module.original_max_position_embeddings, self.orig_module.beta_fast, self.orig_module.beta_slow, self.orig_module.mscale, self.orig_module.mscale_all_dim)

class YarnRotaryEmbeddingV3(BaseInjectedModule):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, generate_device: str='cuda', prefill_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        kwargs = {key: self.config.rope_scaling[key] for key in ['original_max_position_embeddings', 'beta_fast', 'beta_slow', 'mscale', 'mscale_all_dim'] if key in self.config.rope_scaling}
        self._init(dim=self.config.qk_rope_head_dim, max_position_embeddings=self.config.max_position_embeddings, base=self.config.rope_theta, device=self.device, scaling_factor=self.config.rope_scaling['factor'], **kwargs)

    @maybe_no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self._mscale
            sin = emb.sin() * self._mscale
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

    def _init(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, original_max_position_embeddings=4096, beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0):
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        freq_extra = 1.0 / self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        low, high = yarn_find_correction_range(self.beta_fast, self.beta_slow, dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        self.inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self._mscale = float(yarn_get_mscale(self.scaling_factor, self.mscale) / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
        self.max_seq_len_cached = max_position_embeddings

class RotaryEmbeddingV4(BaseInjectedModule):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, generate_device: str='cuda', prefill_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    @maybe_no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

    def load(self):
        self._init(dim=self.config.qk_rope_head_dim, max_position_embeddings=self.config.max_position_embeddings, base=self.config.rope_theta, device=self.device)

    def _init(self, dim, max_position_embeddings, base, device, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        self.max_seq_len_cached = max_position_embeddings