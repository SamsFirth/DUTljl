from functools import wraps
import torch, yaml, pathlib
import os, sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)
from joyaiSFT.util.globals import GLOBAL_CONFIG

def maybe_no_grad(_func=None):

    def decorator(func):

        def wrapper(*args, **kwargs):
            if GLOBAL_CONFIG._config['mod'] == 'sft':
                return func(*args, **kwargs)
            elif GLOBAL_CONFIG._config['mod'] == 'infer':
                with torch.no_grad():
                    return func(*args, **kwargs)
        return wrapper
    if _func is None:
        return decorator
    else:
        return decorator(_func)