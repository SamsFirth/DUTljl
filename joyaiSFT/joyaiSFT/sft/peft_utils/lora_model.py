from __future__ import annotations
from abc import ABC
import math
import operator
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from typing import Literal, Optional, Union
import logging
import torch
from torch import nn
from tqdm import tqdm
from peft.utils.other import get_pattern_key
from peft.utils import ModulesToSaveWrapper, _get_submodules
from peft.tuners.tuners_utils import check_target_module_exists
from peft.config import PeftConfig
from joyaiSFT.sft.peft_utils.lora_layer import dispatch_default, LoraLayer
logger = logging.getLogger(__name__)

class LoraModel(nn.Module, ABC):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """
    prefix: str = 'lora_'

    def __init__(self, model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool=False) -> None:
        super().__init__()
        self.model = model
        self.targeted_module_names: list[str] = []
        if not hasattr(self, 'peft_config'):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info('Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!')
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                self.peft_config.update(peft_config)
        self.active_adapter: str | list[str] = adapter_name
        self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
        self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        self.model.peft_config = self.peft_config

    def inject_adapter(self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool=True, low_cpu_mem_usage: bool=False) -> None:
        """
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the loading process.

        """
        peft_config = self.peft_config[adapter_name]
        excluded_modules = []
        unmatched_modules = []
        _check_for_modules_to_save = getattr(peft_config, 'modules_to_save', None) is not None
        _has_modules_to_save = False
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            if not key:
                continue
            if _check_for_modules_to_save and any((key.endswith(f'{module_to_save}') for module_to_save in peft_config.modules_to_save)):
                parent, target, target_name = _get_submodules(model, key)
                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)
                _has_modules_to_save = True
                continue
            result = check_target_module_exists(peft_config, key)
            if not result:
                unmatched_modules.append(key)
            else:
                self.targeted_module_names.append(key)
                parent, target, target_name = _get_submodules(model, key)
                self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)
        self._mark_only_adapters_as_trainable(model)
        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

    def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key):
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)
        kwargs = {'r': r, 'lora_alpha': alpha, 'lora_dropout': lora_config.lora_dropout, 'fan_in_fan_out': lora_config.fan_in_fan_out, 'init_lora_weights': lora_config.init_lora_weights, 'use_rslora': lora_config.use_rslora, 'use_dora': lora_config.use_dora, 'ephemeral_gpu_offload': lora_config.runtime_config.ephemeral_gpu_offload, 'lora_bias': lora_config.lora_bias, 'loaded_in_8bit': getattr(self.model, 'is_loaded_in_8bit', False), 'loaded_in_4bit': getattr(self.model, 'is_loaded_in_4bit', False)}
        new_module = self._create_new_module(lora_config, adapter_name, target, parent, **kwargs)
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, 'orig_module'):
            child = child.orig_module
        if not hasattr(new_module, 'orig_module'):
            if hasattr(new_module, 'W_q'):
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, 'bias'):
                new_module.bias = child.bias
        if getattr(child, 'state', None) is not None:
            if hasattr(new_module, 'orig_module'):
                new_module.orig_module.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
        meta = torch.device('meta')
        for name, module in new_module.named_modules():
            if self.prefix in name or 'ranknum' in name:
                weight = child.qweight if hasattr(child, 'qweight') else child.W_q if hasattr(child, 'W_q') else child.weight if hasattr(child, 'weight') else child.generate_linear.weight if hasattr(child.generate_linear, 'weight') else next(child.parameters())
                if not any((p.device == meta for p in module.parameters())):
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == 'none':
                continue
            if bias == 'all':
                for n, p in model.named_parameters():
                    if 'bias' in n:
                        p.requires_grad = True
            elif bias == 'lora_only':
                for m in model.modules():
                    if isinstance(m, LoraLayer) and hasattr(m, 'bias') and (m.bias is not None):
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f'Requested bias: {bias}, is not implemented.')

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, parent, **kwargs):
        dispatchers = []
        dispatchers.extend([dispatch_default])
        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target=target, adapter_name=adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:
                break
        if new_module is None:
            raise ValueError(f'Target module {target} is not supported. Currently, only the following modules are supported: `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `transformers.pytorch_utils.Conv1D`.')
        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'model':
                raise
            return getattr(self.model, name)

    def _pre_injection_hook(self, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        """
        A hook to be called before the adapter is injected into the model. This method can be overridden by child
        classes to perform any pre-injection operations.

        Args:
            model (`nn.Module`):
                The model to be adapted.
            config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
        """
        pass

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        return self.active_adapter