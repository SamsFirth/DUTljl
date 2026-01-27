"""
Emitter for Sm90 Epilogue Visitor
"""
from cutlass_library import DataTypeTag, EpilogueScheduleTag
from cutlass.backend import GemmOperationUniversal
from cutlass.backend.evt.backend.emitter_base import FusionCallbacks

class CollectiveEpilogue:

    def __init__(self, tile_description, schedule, element_c, element_d, fusion_callbacks) -> None:
        self.cta_tile_mnk = tile_description.threadblock_shape
        self.element_c = element_c
        self.element_d = element_d
        self.schedule = schedule
        self.fusion_callbacks = fusion_callbacks

    @property
    def CtaTileMNK(self) -> str:
        """
        The threadblock shape
        """
        return f'cute::Shape<_{self.cta_tile_mnk[0]}, _{self.cta_tile_mnk[1]}, _{self.cta_tile_mnk[2]}>'

    @property
    def EpilogueTileType(self) -> str:
        """
        The epilogue tile type
        """
        return 'cutlass::epilogue::collective::EpilogueTileAuto'

    @property
    def Schedule(self) -> str:
        return EpilogueScheduleTag[self.schedule]

    def emit(self):
        callback_decl, callback_name = self.fusion_callbacks.emit()
        return (callback_name, f'\nusing EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<\n  {self.CtaTileMNK}, {self.EpilogueTileType},\n  {DataTypeTag[self.element_c]}, {DataTypeTag[self.element_d]},\n  {self.Schedule}\n>;\n{callback_decl}\n')

class Sm90Emitter:

    def __init__(self, operation: GemmOperationUniversal, graph) -> None:
        fusion_callbacks = FusionCallbacks(graph, cc=90, emit_CD=False)
        self.collective_epilogue = CollectiveEpilogue(tile_description=operation.tile_description, schedule=operation.tile_description.epilogue_schedule, element_c=operation.C.element, element_d=operation.C.element, fusion_callbacks=fusion_callbacks)

    def emit(self):
        return self.collective_epilogue.emit()