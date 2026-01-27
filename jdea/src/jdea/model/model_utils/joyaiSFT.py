import importlib.util as _u
from typing import TYPE_CHECKING, Any
import torch
from ...extras import logging
from ...extras.misc import get_current_device
if TYPE_CHECKING:
    from ...hparams import FinetuningArguments, ModelArguments
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
JOY_AVAILABLE = _u.find_spec('joyaiSFT') is not None
if JOY_AVAILABLE:
    from joyaiSFT.models.modeling_chatrhinoV0 import ChatrhinoV0ForCausalLM
    from joyaiSFT.models.modeling_chatrhino import ChatrhinoForCausalLM
    from joyaiSFT.optimize.optimize import optimize_and_load_gguf
    from joyaiSFT.server.config.config import Config
    from joyaiSFT.sft.lora import inject_lora_layer
    from joyaiSFT.util.custom_loader import GGUFLoader, SafeTensorLoader
    from joyaiSFT.util.globals import GLOBAL_CONFIG
    from joyaiSFT.util.utils import load_weights
logger = logging.get_logger(__name__)

from jdea.hparams.optimize_rule import resolve_joy_optimize_rule_text

from jdea.hparams.model_config_resolve import resolve_chatrhino_config

def _get_joy_kwargs(config: 'PretrainedConfig', model_name_or_path: str, model_args: 'ModelArguments', finetuning_args: 'FinetuningArguments') -> dict[str, Any]:
    return {'model_name': model_name_or_path, 'max_seq_length': model_args.model_max_length or 4096, 'dtype': model_args.compute_dtype, 'load_in_4bit': model_args.quantization_bit == 4, 'token': model_args.hf_hub_token, 'full_finetuning': finetuning_args.finetuning_type == 'full', 'device_map': {'': get_current_device()}, 'rope_scaling': getattr(config, 'rope_scaling', None), 'fix_tokenizer': False, 'trust_remote_code': model_args.trust_remote_code, 'use_gradient_checkpointing': 'joyaiSFT'}

def load_joy_pretrained_model(config: 'PretrainedConfig', model_args: 'ModelArguments') -> 'PreTrainedModel':
    """Optionally load pretrained model with JOYSFT. Used in training."""
    custom_models = {'ChatrhinoV0ForCausalLM': ChatrhinoV0ForCausalLM, 'ChatrhinoForCausalLM': ChatrhinoForCausalLM}
    Config().cpu_infer = model_args.cpu_infer
    Config().chunk_size = model_args.chunk_size
    config = resolve_chatrhino_config()
    if model_args.mode == 'long_context':
        assert config.architectures[0] == 'LlamaForCausalLM', 'only LlamaForCausalLM support long_context mode'
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)
    with torch.device('meta'):
        if config.architectures[0] in custom_models:
            print('using custom modeling_xxx.py.')
            if 'Qwen2Moe' in config.architectures[0]:
                config._attn_implementation = 'flash_attention_2'
            if 'Llama' in config.architectures[0]:
                config._attn_implementation = 'eager'
            if 'Mixtral' in config.architectures[0]:
                config._attn_implementation = 'flash_attention_2'
            model = custom_models[config.architectures[0]](config)
        else:
            attn_implementation = 'flash_attention_2'
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation=attn_implementation)
    rule_text = resolve_joy_optimize_rule_text(model_args.joy_optimize_rule)
    gguf_path = model_args.model_name_or_path

    assert rule_text is not None and rule_text != "", "rule_text must be provided (YAML text)."
    assert gguf_path is not None, "gguf_path must be provided (path to a folder or .gguf file)."

    GLOBAL_CONFIG._config["mod"] = "infer"
    optimize_and_load_gguf(model, rule_text, gguf_path, config)

    return model

def get_joy_peft_model(model: 'PreTrainedModel', peft_kwargs: dict[str, Any]) -> 'PreTrainedModel':
    """Get the peft model for the pretrained model with JOYSFT. Used in training."""
    from joyaiSFT.sft.peft_utils.mapping import get_peft_model
    return get_peft_model(model, peft_kwargs)

def load_joy_peft_model(model_args: 'ModelArguments', model: 'PreTrainedModel') -> 'PreTrainedModel':
    """Load peft model with JOYSFT. Used in both training and inference."""
    load_adapter_name_or_path = model_args.adapter_name_or_path[0]
    if load_adapter_name_or_path.endswith('.gguf'):
        inject_lora_layer(model, load_adapter_name_or_path)
        adapter_gguf_loader = GGUFLoader(load_adapter_name_or_path)
        load_weights(model, adapter_gguf_loader, adapter_gguf=True)
        model.train()
    else:
        inject_lora_layer(model, load_adapter_name_or_path)
        adapter_loader = SafeTensorLoader(load_adapter_name_or_path)
        device = next(model.parameters()).device
        for key in adapter_loader.tensor_file_map.keys():
            try:
                tensor = adapter_loader.load_tensor(key, device=device)
                model_key = key.replace('base_model.model.', '')
                model_key = model_key.replace('.weight', '.default.weight')
                model_key = model_key.replace('.default.default.weight', '.default.weight')
                param = model.get_parameter(model_key)
                param.data.copy_(tensor.data)
                print(f'Loaded adapter weight: {key} -> {model_key}')
            except AttributeError:
                print(f'Skipping {key}: not a model parameter')
            except KeyError:
                print(f'Key not found in model: {model_key} (original: {key})')
    return model