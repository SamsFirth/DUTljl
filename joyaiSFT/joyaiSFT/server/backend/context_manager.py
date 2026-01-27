from asyncio import Lock
from typing import Dict, Optional
from joyaiSFT.server.backend.base import ThreadContext, BackendInterfaceBase
from joyaiSFT.server.schemas.assistants.runs import RunObject
from joyaiSFT.server.schemas.base import ObjectID
from joyaiSFT.server.config.log import logger
from joyaiSFT.server.backend.interfaces.transformers import TransformersThreadContext
from joyaiSFT.server.backend.interfaces.joyaiSFT import JOYSFTThreadContext
from joyaiSFT.server.backend.interfaces.exllamav2 import ExllamaThreadContext
from joyaiSFT.server.backend.interfaces.exllamav2 import ExllamaInterface
from joyaiSFT.server.backend.interfaces.transformers import TransformersInterface
from joyaiSFT.server.backend.interfaces.joyaiSFT import JOYSFTInterface

class ThreadContextManager:
    lock: Lock
    threads_context: Dict[ObjectID, ThreadContext]
    interface: BackendInterfaceBase

    def __init__(self, interface) -> None:
        logger.debug(f'Creating Context Manager')
        self.lock = Lock()
        self.threads_context = {}
        self.interface = interface
        pass

    async def get_context_by_run_object(self, run: RunObject) -> ThreadContext:
        async with self.lock:
            logger.debug(f'keys {self.threads_context.keys()}')
            if run.thread_id not in self.threads_context:
                logger.debug(f'new inference context {run.thread_id}')
                if isinstance(self.interface, ExllamaInterface):
                    new_context = ExllamaThreadContext(run, self.interface)
                elif isinstance(self.interface, JOYSFTInterface):
                    new_context = JOYSFTThreadContext(run, self.interface)
                elif isinstance(self.interface, TransformersInterface):
                    new_context = TransformersThreadContext(run, self.interface)
                else:
                    from joyaiSFT.server.backend.interfaces.balance_serve import BalanceServeThreadContext
                    from joyaiSFT.server.backend.interfaces.balance_serve import BalanceServeInterface
                    if isinstance(self.interface, BalanceServeInterface):
                        new_context = BalanceServeThreadContext(run, self.interface)
                    else:
                        raise NotImplementedError
                self.threads_context[run.thread_id] = new_context
            re = self.threads_context[run.thread_id]
            re.update_by_run(run)
            return re

    async def get_context_by_thread_id(self, thread_id: ObjectID) -> Optional[ThreadContext]:
        async with self.lock:
            if thread_id in self.threads_context:
                logger.debug(f'found context for thread {thread_id}')
                return self.threads_context[thread_id]
            else:
                logger.debug(f'no context for thread {thread_id}')
                return None