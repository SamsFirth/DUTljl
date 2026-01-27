from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import os
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.operators.linear import JOYSFTLinear
from joyaiSFT.util.custom_loader import GGUFLoader, ModelLoader, SafeTensorLoader
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod

class JOYMoEGateBase(ABC):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str='cuda', **kwargs):
        super().__init__()
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
        self.orig_module = orig_module

    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str='cpu', warmup: bool=False):
        pass

    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None=None, device: str='cpu'):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None
        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_gate(key, device=device)
            elif self.gguf_loader.has_tensor(key + '.weight'):
                targets = ['.weight', '.e_score_correction_bias']
                tensors = self.load_multi(key, targets, device=device)
                weight = tensors['.weight']
                e_score_correction_bias = tensors['.e_score_correction_bias']
                res = {'weight': weight, 'e_score_correction_bias': e_score_correction_bias}
            else:
                raise ValueError(f'Experts {key} not found in gguf_loader')
        return res

    def load_multi(self, key: str, keys: list[str], device: str='cpu'):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors

class JOYMoEGate(BaseInjectedModule, JOYMoEGateBase):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module=None, generate_device: str='cuda', prefill_device: str='cuda', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        JOYMoEGateBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def forward(self, hidden_states) -> torch.Tensor:
        return self.orig_module.forward(hidden_states)

    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str | None=None):
        if device is None:
            device = self.device
        if w is None:
            w = self.load_weights(device=device)
        if isinstance(w, dict):
            self.orig_module.weight = nn.Parameter(w['weight'])
            self.orig_module.e_score_correction_bias = nn.Parameter(w['e_score_correction_bias'])
        else:
            raise ValueError('Invalid weight type')
        self.orig_module.weight = nn.Parameter(self.orig_module.weight.to(device))
        self.orig_module.e_score_correction_bias = nn.Parameter(self.orig_module.e_score_correction_bias.to(device))

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = None

class JOYMoEGateIPEXLLM(JOYMoEGate):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module=None, generate_device: str='xpu', prefill_device: str='xpu', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        JOYMoEGate.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def forward(self, hidden_states) -> torch.Tensor:
        x = hidden_states.view(-1, hidden_states.size(-1))
        logits = torch.nn.functional.linear(x.type(torch.float32), self.orig_module.weight.type(torch.float32), None)
        scores = logits.sigmoid()
        from ipex_llm.transformers.models.common import moe_group_topk
        topk_idx, topk_weight = moe_group_topk(scores, self.orig_module.e_score_correction_bias, self.n_group, self.topk_group, self.top_k, self.norm_topk_prob, self.routed_scaling_factor)
        return (topk_idx, topk_weight.to(x.dtype))