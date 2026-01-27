"""
Emitter for Sm80 Epilogue Visitor
"""
from cutlass.backend.evt.backend.emitter_base import FusionCallbacks
from cutlass.backend import GemmOperationUniversal

class Sm80Emitter:

    def __init__(self, operation: GemmOperationUniversal, graph) -> None:
        self.fusion_callbacks = FusionCallbacks(graph, cc=80)

    def emit(self):
        callback_decl, callback_name = self.fusion_callbacks.emit()
        return (callback_name, callback_decl)