# jdea/hparams/optimize_rule.py
from typing import Optional
from .optimize_rules_inline import DEFAULT_OPT_RULE_YAML

def resolve_joy_optimize_rule_text(rule: Optional[str]) -> str:
    """
    返回 YAML 文本：
      - None / "" / "default"：返回内置默认 YAML 文本
      - 其他：按 YAML 文本处理（即用户可以直接把 YAML 文本传进参数）
    """
    if rule is None or rule == "" or rule == "default":
        return DEFAULT_OPT_RULE_YAML
    return rule
