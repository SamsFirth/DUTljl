import logging
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig
import torch
import torch.nn as nn
from joyaiSFT.models.modeling_chatrhino import ChatrhinoRMSNorm
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.util.custom_loader import GGUFLoader
if not torch.xpu.is_available():
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm
logger = logging.getLogger(__name__)

class RMSNorm(ChatrhinoRMSNorm, BaseInjectedModule):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', generate_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.hidden_size, orig_module.variance_epsilon)

    def forward(self, x: torch.Tensor, batch_size_tensor: torch.Tensor=None, residual: Optional[torch.Tensor]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if batch_size_tensor is None:
            return self.forward_native(x)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            return (x, residual)
        out = rmsnorm(x, self.weight.data, batch_size_tensor, self.variance_epsilon)
        return out

    def forward_native(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class ChatrhinoRMSNormTorch(ChatrhinoRMSNorm, BaseInjectedModule):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', generate_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.hidden_size, orig_module.variance_epsilon)

    def forward(self, x, batch_size_tensor: torch.Tensor=None, residual: Optional[torch.Tensor]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if residual is not None:
            return (self.weight * x.to(input_dtype), residual)
        return self.weight * x.to(input_dtype)

class JOYChatrhinoV0RMSNormIPEXLLM(ChatrhinoRMSNorm, BaseInjectedModule):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='xpu', generate_device: str='xpu', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.weight.shape[0], orig_module.variance_epsilon)
        self.eps = orig_module.variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ipex_llm.transformers.models.common import rms_norm_forward
        if x.dtype not in [torch.float32, torch.float16]:
            output = rms_norm_forward(self, x.float())
        else:
            output = rms_norm_forward(self, x)
        return output.to(x.dtype)

    def load(self):
        BaseInjectedModule.load(self)
        if self.weight.dtype not in [torch.float32, torch.float16]:
            self.weight = self.weight.float()