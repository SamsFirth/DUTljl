# src/jdea/hparams/model_config_resolve.py
from __future__ import annotations
from typing import Any, Dict, Optional

from .model_config_inline import DEFAULT_CHATRHINO_CONFIG, _normalize_torch_dtype

def resolve_chatrhino_config(override: Optional[Dict[str, Any]] = None):
    # 这里用你自定义模型的 Config 类（按你的包路径调整）
    from joyaiSFT.models.configuration_chatrhino import ChatrhinoConfig

    cfg = dict(DEFAULT_CHATRHINO_CONFIG)
    if override:
        cfg.update(override)

    cfg = _normalize_torch_dtype(cfg)

    # 一般 PretrainedConfig 会吞掉多余 kwargs；如果你的 ChatrhinoConfig 很严格，再做字段过滤
    return ChatrhinoConfig(**cfg)
