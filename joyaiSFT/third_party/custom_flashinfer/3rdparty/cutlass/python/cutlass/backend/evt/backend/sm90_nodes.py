from pycute import product
from cutlass_library import DataTypeSize, DataTypeTag
from cutlass.backend.evt.ir import AccumulatorImpl, AuxLoadImpl, ColumnBroadcastImpl, LoadNode, LoadSrcImpl, RowBroadcastImpl, ScalarBroadcastImpl, ComputeImpl, ComputeNode, AuxStoreImpl, ColumnReductionImpl, RowReductionImpl, ScalarReductionImpl, StoreNode, StoreDImpl
from cutlass.backend.library import FloatRoundStyleTag, FunctionalOp, op_tag

class Sm90AccumulatorImpl(AccumulatorImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90AccFetch;\n'
        return self._type_decl

class Sm90LoadSrcImpl(LoadSrcImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing ElementC = {DataTypeTag[self.element]};\nusing StrideC = {self.stride_mnl};\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90SrcFetch<{DataTypeTag[self.element]}>;\n'
        return self._type_decl

class Sm90AuxLoadImpl(AuxLoadImpl):

    @property
    def descriptor(self) -> str:
        """
        Descriptor for Aux Load
        """
        return f'{self.name_camel}Descriptor'

    def decl_descriptor(self) -> str:
        """
        Declare the descriptor type
        """
        return f'\nusing {self.descriptor} = cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, {self.stride_mnl}, {DataTypeTag[self.element]}>;\n'

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = self.decl_descriptor()
        self._type_decl += f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90AuxLoad<\n    {self.descriptor}::Stages, typename {self.descriptor}::EpilogueTile, {DataTypeTag[self.element]},\n    {self.stride_mnl}, typename {self.descriptor}::SmemLayoutAtom, typename {self.descriptor}::CopyOpS2R\n>;\n'
        return self._type_decl

    def get_smem_size(self, cta_tile_mnk, epilogue_tile_mn, stages_c, stages_d, epi_tiles):
        """
        Get the shared memory size based on epilogue_tile_mn, stages_c, and stages_d
        """
        return (DataTypeSize[self.element] * stages_c * product(epilogue_tile_mn) // 8, 128)

class Sm90ScalarBroadcastImpl(ScalarBroadcastImpl):

    def __init__(self, node: LoadNode) -> None:
        super().__init__(node)
        self.broadcast_count = 1
        self.reduction_fn = FunctionalOp.Multiplies

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90ScalarBroadcast<\n    {DataTypeTag[self.element]}, {self.stride_mnl}, {self.broadcast_count}, {op_tag(self.reduction_fn)}\n>;\n'
        return self._type_decl

class Sm90RowBroadcastImpl(RowBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90RowBroadcast<\n    0 /*Stages*/, typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]}, {DataTypeTag[self.element_output]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm90ColumnBroadcastImpl(ColumnBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90ColBroadcast<\n    0 /*Stages*/, typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]}, {DataTypeTag[self.element_output]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm90ComputeImpl(ComputeImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90Compute<\n    {op_tag(self.fn)}, {DataTypeTag[self.element_output]}, {DataTypeTag[self.element_compute]},\n    {FloatRoundStyleTag[self.round_style]}\n>;\n'
        return self._type_decl

class Sm90AuxStoreImpl(AuxStoreImpl):

    @property
    def descriptor(self) -> str:
        """
        Descriptor for Aux Load
        """
        return f'{self.name_camel}Descriptor'

    def decl_descriptor(self) -> str:
        """
        Declare the descriptor type
        """
        return f'\nusing {self.descriptor} = cutlass::epilogue::collective::detail::AuxStoreDescriptor<\n    EpilogueDescriptor, {self.stride_mnl}, {DataTypeTag[self.element]}\n>;\n'

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = self.decl_descriptor()
        self._type_decl += f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90AuxStore<\n    {self.descriptor}::Stages, typename {self.descriptor}::EpilogueTile, {DataTypeTag[self.element]},\n    {FloatRoundStyleTag[self.round_style]}, {self.stride_mnl}, typename {self.descriptor}::SmemLayoutAtom,\n    typename {self.descriptor}::CopyOpR2S\n>;\n'
        return self._type_decl

    def get_smem_size(self, cta_tile_mnk, epilogue_tile_mn, stages_c, stages_d, epi_tiles):
        """
        Get the shared memory size based on epilogue_tile_mn, stages_c, and stages_d
        """
        return (DataTypeSize[self.element] * stages_d * product(epilogue_tile_mn) // 8, 128)

class Sm90StoreDImpl(StoreDImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        return f'\nusing ElementD = {DataTypeTag[self.element]};\nusing StrideD = {self.stride_mnl};\n'

class Sm90ColumnReductionImpl(ColumnReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90ColReduction<\n    {op_tag(self.reg_reduce_fn)}, {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)}, 0,\n    typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]},\n    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm90RowReductionImpl(RowReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90RowReduction<\n    {op_tag(self.reg_reduce_fn)}, {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)}, 0 /* Stages */,\n    typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]},\n    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm90ScalarReductionImpl(ScalarReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90ScalarReduction<\n    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},\n    {DataTypeTag[self.element]}, {DataTypeTag[self.element_compute]},\n    {FloatRoundStyleTag[self.round_style]}, {self.stride_mnl}\n>;\n'
        return self._type_decl