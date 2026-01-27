from typing import TYPE_CHECKING, Union
import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations import is_deepspeed_zero3_enabled
from ...extras.misc import check_version
from ...extras.packages import is_transformers_version_greater_than
if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel
    from ...hparams import ModelArguments
if is_transformers_version_greater_than('4.57.0'):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

def _set_z3_leaf_modules(model: 'PreTrainedModel', leaf_modules: list[Union['nn.Module', str]]) -> None:
    check_version('deepspeed>=0.13.0')
    from deepspeed.utils import set_z3_leaf_modules
    set_z3_leaf_modules(model, leaf_modules)

def add_z3_leaf_module(model: 'PreTrainedModel') -> None:
    """Set module as a leaf module to skip partitioning in deepspeed zero3."""
    if not is_deepspeed_zero3_enabled():
        return
    model_type = getattr(model.config, 'model_type', None)
    text_config = getattr(model.config, 'text_config', None)
    text_model_type = getattr(text_config, 'model_type', None)
    if model_type == 'dbrx':
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN
        _set_z3_leaf_modules(model, [DbrxFFN])
    if model_type == 'chatrhinoV0':
        _set_z3_leaf_modules(model, ['ChatrhinoV0MoE'])
    if model_type == 'chatrhino':
        _set_z3_leaf_modules(model, ['ChatrhinoMoE'])
    if model_type == 'ernie4_5_moe':
        from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeSparseMoeBlock
        _set_z3_leaf_modules(model, [Ernie4_5_MoeSparseMoeBlock])
    if model_type == 'granitemoe':
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE
        _set_z3_leaf_modules(model, [GraniteMoeMoE])
    if model_type == 'glm4_moe':
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
        _set_z3_leaf_modules(model, [Glm4MoeMoE])
    if model_type == 'glm4v_moe':
        from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextMoE
        _set_z3_leaf_modules(model, [Glm4vMoeTextMoE])
    if model_type == 'jamba':
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock
        _set_z3_leaf_modules(model, [JambaSparseMoeBlock])
    if model_type == 'jetmoe':
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE
        _set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])
    if model_type == 'llama4':
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
        _set_z3_leaf_modules(model, [Llama4TextMoe])
    if model_type == 'mixtral':
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        _set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    if model_type == 'olmoe':
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
        _set_z3_leaf_modules(model, [OlmoeSparseMoeBlock])
    if model_type == 'phimoe':
        from transformers.models.phimoe.modeling_phimoe import PhimoeSparseMoeBlock
        _set_z3_leaf_modules(model, [PhimoeSparseMoeBlock])
    if model_type == 'qwen2_moe':
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
        _set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
    if model_type == 'qwen3_moe' or text_model_type == 'qwen3_moe':
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        _set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])
    if model_type == 'qwen3_vl_moe':
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
        _set_z3_leaf_modules(model, [Qwen3VLMoeTextSparseMoeBlock])
    if model_type in ('qwen3_omni_moe', 'qwen3_omni_moe_thinker'):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock
        _set_z3_leaf_modules(model, [Qwen3OmniMoeThinkerTextSparseMoeBlock])

def configure_moe(config: 'PretrainedConfig', model_args: 'ModelArguments', is_trainable: bool) -> None:
    if not is_trainable or not model_args.moe_aux_loss_coef:
        return
    model_type = getattr(config, 'model_type', None)
    text_config = getattr(config, 'text_config', None)
    if model_type in ['dbrx', 'ernie4_5_moe', 'granitemoe', 'jamba', 'jetmoe', 'llama4', 'mixtral', 'olmoe', 'phimoe', 'qwen2_moe', 'qwen3_moe']:
        setattr(config, 'output_router_logits', True)
    if text_config and getattr(text_config, 'model_type', None) in ['glm4v_moe_text', 'qwen3_moe']:
        setattr(text_config, 'output_router_logits', True)
    if model_type in ['ernie4_5_moe', 'granitemoe', 'jamba', 'llama4', 'mixtral', 'olmoe', 'phimoe', 'qwen2_moe', 'qwen3_moe']:
        setattr(config, 'router_aux_loss_coef', model_args.moe_aux_loss_coef)
    elif text_config and getattr(text_config, 'model_type', None) in ['qwen3_moe']:
        setattr(text_config, 'router_aux_loss_coef', model_args.moe_aux_loss_coef)
    elif model_type == 'chatrhino':
        setattr(config, 'aux_loss_alpha', model_args.moe_aux_loss_coef)
    elif model_type == 'jetmoe':
        setattr(config, 'aux_loss_coef', model_args.moe_aux_loss_coef)

class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        full_routing_weights = torch.zeros_like(routing_weights)
        full_routing_weights.scatter_(1, top_k_indices, top_k_weights)
        if self.norm_topk_prob:
            top_k_sum = full_routing_weights.sum(dim=-1, keepdim=True)
            top_k_sum = torch.clamp(top_k_sum, min=1e-09)
            full_routing_weights /= top_k_sum
        full_routing_weights = full_routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_weights = full_routing_weights[:, expert_idx, None]
            current_hidden_states = expert_layer(hidden_states) * expert_weights
            final_hidden_states += current_hidden_states
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return (final_hidden_states, router_logits)