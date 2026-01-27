from typing import Any
from torch import nn, Tensor
from joyaiSFT.util.custom_loader import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
import joyaiSFT.util.utils as utils

class BaseInjectedModule(nn.Module):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', generate_device: str='cuda', **kwargs):
        nn.Module.__init__(self)
        nn.Module.__setattr__(self, 'orig_module', orig_module)
        object.__setattr__(self, 'key', key)
        object.__setattr__(self, 'gguf_loader', gguf_loader)
        object.__setattr__(self, 'config', config)
        object.__setattr__(self, 'prefill_device', prefill_device)
        object.__setattr__(self, 'generate_device', generate_device)
        object.__setattr__(self, 'device', generate_device)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except:
            if name == 'orig_module':
                return nn.Module.__getattr__(self, 'orig_module')
            try:
                return nn.Module.__getattr__(self, 'orig_module').__getattr__(name)
            except:
                return super(nn.Module, nn.Module.__getattr__(self, 'orig_module')).__getattribute__(name)

    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:
        if name == 'orig_module':
            return nn.Module.__setattr__(self, 'orig_module', value)
        elif hasattr(self, name):
            return object.__setattr__(self, name, value)
        return nn.Module.__getattr__(self, 'orig_module').__setattr__(name, value)

    def forward(self, *args, **kwargs):
        return self.orig_module.forward(*args, **kwargs)

    def load(self, gguf_loader=None, adapter_gguf: bool=False):
        for name, child in self._modules.items():
            if gguf_loader == None:
                utils.load_weights(child, self.gguf_loader, self.key + '.', adapter_gguf=adapter_gguf)
            else:
                utils.load_weights(child, gguf_loader, self.key + '.', adapter_gguf=adapter_gguf)