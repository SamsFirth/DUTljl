"""
Description  :  
Author       : qiyuxinlin
Date         : 2024-07-25 11:50:16
Version      : 1.0.0
LastEditors  : qiyuxinlin 
LastEditTime : 2024-07-25 12:54:48
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""
from joyaiSFT.server.config.config import Config
from joyaiSFT.server.backend.args import ConfigArgs
from joyaiSFT.server.backend.context_manager import ThreadContextManager
from joyaiSFT.server.backend.interfaces.exllamav2 import ExllamaInterface
from joyaiSFT.server.backend.interfaces.transformers import TransformersInterface
from joyaiSFT.server.backend.interfaces.joyaiSFT import JOYSFTInterface

def create_interface(config: Config, default_args: ConfigArgs):
    if config.backend_type == 'transformers':
        from joyaiSFT.server.backend.interfaces.transformers import TransformersInterface as BackendInterface
    elif config.backend_type == 'exllamav2':
        from joyaiSFT.server.backend.interfaces.exllamav2 import ExllamaInterface as BackendInterface
    elif config.backend_type == 'joyaiSFT':
        from joyaiSFT.server.backend.interfaces.joyaiSFT import JOYSFTInterface as BackendInterface
    elif config.backend_type == 'balance_serve':
        from joyaiSFT.server.backend.interfaces.balance_serve import BalanceServeInterface as BackendInterface
    else:
        raise NotImplementedError(f'{config.backend_type} not implemented')
    GlobalInterface.interface = BackendInterface(default_args)
    GlobalContextManager.context_manager = ThreadContextManager(GlobalInterface.interface)

class GlobalContextManager:
    context_manager: ThreadContextManager

class GlobalInterface:
    interface: TransformersInterface | JOYSFTInterface | ExllamaInterface

def get_thread_context_manager() -> GlobalContextManager:
    return GlobalContextManager.context_manager

def get_interface() -> GlobalInterface:
    return GlobalInterface.interface