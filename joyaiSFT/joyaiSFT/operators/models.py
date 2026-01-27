import inspect
import math
from typing import List, Optional, Tuple, Union
import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from joyaiSFT.operators.dynamic_attention import DynamicScaledDotProductAttention
from joyaiSFT.server.config.config import Config
import os
import yaml
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast, SequenceClassifierOutputWithPast, TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging, replace_return_docstrings
from joyaiSFT.models.modeling_chatrhinoV0 import BaseModelOutputWithPast, ChatrhinoV0DecoderLayer, ChatrhinoV0MoE
from joyaiSFT.util.vendors import device_manager, get_device, to_device, GPUVendor
from joyaiSFT.operators.base_operator import BaseInjectedModule
from joyaiSFT.util.inference_state import InferenceState
from joyaiSFT.util.utils import get_compute_capability
from joyaiSFT.util.custom_loader import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    _flash_supports_window_size = 'window_size' in list(inspect.signature(flash_attn_func).parameters)
logger = logging.get_logger(__name__)
ChatrhinoV0_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see\n            `past_key_values`).\n\n            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]\n            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more\n            information on the default strategy.\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.n_positions - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):\n            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`\n            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.\n\n            Two formats are allowed:\n            - a [`~cache_utils.Cache`] instance;\n            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy\n            cache format.\n\n            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the\n            legacy cache format will be returned.\n\n            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't\n            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`\n            of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class JOYChatrhinoV0Model(BaseInjectedModule):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ChatrhinoV0DecoderLayer`]

    Args:
        config: ChatrhinoV0Config
    """

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str='cuda', per_layer_prefill_intput_threshold: int=30000, transfer_map: dict=None, **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.per_layer_prefill_intput_threshold = per_layer_prefill_intput_threshold
        self.transfer_map = transfer_map
        self.stream_device_map = dict()

    @add_start_docstrings_to_model_forward(ChatrhinoV0_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, per_layer_prefill_intput_threshold: int | None=None) -> Union[Tuple, BaseModelOutputWithPast]:
        self.gradient_checkpointing = False
        if per_layer_prefill_intput_threshold is None:
            per_layer_prefill_intput_threshold = self.per_layer_prefill_intput_threshold
        per_layer_prefill_flag = False
        seq_lenth = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
        if per_layer_prefill_intput_threshold and per_layer_prefill_intput_threshold < seq_lenth:
            per_layer_prefill_flag = True
            for layer in self.layers:
                self.load_layer_to(layer, InferenceState.UNLOAD)
            torch.cuda.empty_cache()
        else:
            pass
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers.')
                use_cache = False
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        if inputs_embeds is None:
            org_device = input_ids.device
            input_ids = input_ids.to(self.embed_tokens.weight.device)
            inputs_embeds = self.embed_tokens(input_ids).to(org_device)
            input_ids = input_ids.to(org_device)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if inputs_embeds.device.type == 'xpu' and position_ids is not None:
            cos, sin = self.layers[0].self_attn.rotary_emb(inputs_embeds, position_ids)
            position_embeddings = (cos, sin)
        else:
            position_embeddings = None
        if per_layer_prefill_flag:
            causal_mask = None
        elif os.name == 'nt' or get_compute_capability() < 8 or (self.transfer_map is not None and 'cpu' in self.transfer_map.values()) or (device_manager.gpu_vendor != GPUVendor.NVIDIA):
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        else:
            causal_mask = None
        hidden_states = inputs_embeds
        if per_layer_prefill_flag:
            print(f'Total length of input_ids: {hidden_states.size(1)}')
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        t_gpu = 0
        t_cpu = 0
        t_f = 0
        for i, decoder_layer in enumerate(self.layers):
            if self.transfer_map is not None and i in self.transfer_map:
                prev_stream = torch.cuda.current_stream()
                cur_device = self.transfer_map[i]
                if cur_device not in self.stream_device_map and cur_device.lower() != 'cpu':
                    self.stream_device_map[cur_device] = torch.cuda.Stream(cur_device)
                if cur_device.lower() != 'cpu':
                    torch.cuda.set_device(cur_device)
                    self.stream_device_map[cur_device].wait_stream(prev_stream)
                    torch.cuda.set_stream(self.stream_device_map[cur_device])
                hidden_states = hidden_states.to(self.transfer_map[i], non_blocking=True)
                causal_mask = causal_mask.to(self.transfer_map[i], non_blocking=True) if causal_mask is not None else None
                position_ids = position_ids.to(self.transfer_map[i], non_blocking=True) if position_ids is not None else None
                cache_position = cache_position.to(self.transfer_map[i], non_blocking=True) if cache_position is not None else None
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position)
            else:
                t3 = time.time()
                if per_layer_prefill_flag:
                    self.load_layer_to(decoder_layer, InferenceState.PREFILL)
                    torch.cuda.empty_cache()
                t4 = time.time()
                layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                t5 = time.time()
                if per_layer_prefill_flag:
                    self.load_layer_to(decoder_layer, InferenceState.UNLOAD)
                    torch.cuda.empty_cache()
                t6 = time.time()
            t_gpu += t4 - t3
            t_cpu += t6 - t5
            t_f += t5 - t4
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if per_layer_prefill_flag:
            t6 = time.time()
            per_layer_prefill_flag = False
            for layer in self.layers:
                self.load_layer_to(layer, InferenceState.GENERATE)
            torch.cuda.empty_cache()
            t7 = time.time()
            print(f'total time: {t7 - t3}, \n layer num{len(self.layers)}, gpu time: {t_gpu}, cpu time: {t_cpu}, forward time: {t_f}, restore time: {t7 - t6}')
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns)

    def load_layer_to(self, layer: ChatrhinoV0DecoderLayer, target: InferenceState):
        assert isinstance(layer, ChatrhinoV0DecoderLayer), 'module should be nn.ModuleList of decoder layers'
        device = 'cpu' if target == InferenceState.UNLOAD else 'cuda'
        layer.self_attn.to(device)
        if isinstance(layer.mlp, ChatrhinoV0MoE):
            layer.mlp.gate.to(device)
            layer.mlp.experts.set_inference_mode(target)
            layer.mlp.shared_experts.gate_proj.set_inference_mode(target)
            layer.mlp.shared_experts.up_proj.set_inference_mode(target)
            layer.mlp.shared_experts.down_proj.set_inference_mode(target)
            layer.mlp.shared_experts.act_fn.to(device)
        else:
            layer.mlp.gate_proj.set_inference_mode(target)
            layer.mlp.up_proj.set_inference_mode(target)
            layer.mlp.down_proj.set_inference_mode(target)
            layer.mlp.act_fn.to(device)
        layer.input_layernorm.to(device)
        layer.post_attention_layernorm.to(device)