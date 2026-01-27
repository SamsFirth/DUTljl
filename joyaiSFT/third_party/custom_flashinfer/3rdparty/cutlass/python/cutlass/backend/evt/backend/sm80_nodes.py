from cutlass_library import DataTypeSize, DataTypeTag
from cutlass.backend.evt.ir import AccumulatorImpl, AuxLoadImpl, ColumnBroadcastImpl, LoadNode, LoadSrcImpl, RowBroadcastImpl, ScalarBroadcastImpl, ComputeImpl, AuxStoreImpl, ColumnReductionImpl, RowReductionImpl, ScalarReductionImpl
from cutlass.backend.library import FloatRoundStyleTag, FunctionalOp, op_tag

class Sm80AccumulatorImpl(AccumulatorImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorAccFetch;\n'
        return self._type_decl

class Sm80AuxLoadImpl(AuxLoadImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorAuxLoad<\n    OutputTileThreadMap, {DataTypeTag[self.element]}, {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm80LoadSrcImpl(Sm80AuxLoadImpl):
    pass

class Sm80ScalarBroadcastImpl(ScalarBroadcastImpl):

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
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorScalarBroadcast<\n    {DataTypeTag[self.element]}, {self.stride_mnl}, {self.broadcast_count}, {op_tag(self.reduction_fn)}\n>;\n'
        return self._type_decl

class Sm80RowBroadcastImpl(RowBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorRowBroadcast<\n    OutputTileThreadMap, {DataTypeTag[self.element]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm80ColumnBroadcastImpl(ColumnBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorColBroadcast<\n    OutputTileThreadMap, {DataTypeTag[self.element]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm80ComputeImpl(ComputeImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorCompute<\n    {op_tag(self.fn)}, {DataTypeTag[self.element_output]}, {DataTypeTag[self.element_compute]},\n    {FloatRoundStyleTag[self.round_style]}\n>;\n'
        return self._type_decl

class Sm80AuxStoreImpl(AuxStoreImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorAuxStore<\n    OutputTileThreadMap, {DataTypeTag[self.element]}, {FloatRoundStyleTag[self.round_style]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm80StoreDImpl(Sm80AuxStoreImpl):
    pass

class Sm80ColumnReductionImpl(ColumnReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorColReduction<\n    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},\n    OutputTileThreadMap, {DataTypeTag[self.element]},\n    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm80RowReductionImpl(RowReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorRowReduction<\n    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},\n    OutputTileThreadMap, {DataTypeTag[self.element]},\n    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl

class Sm80ScalarReductionImpl(ScalarReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl
        self._type_decl = f'\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorScalarReduction<\n    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},\n    OutputTileThreadMap, {DataTypeTag[self.element]},\n    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},\n    {self.stride_mnl}\n>;\n'
        return self._type_decl