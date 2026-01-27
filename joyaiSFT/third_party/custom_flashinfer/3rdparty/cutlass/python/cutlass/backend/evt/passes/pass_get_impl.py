"""
Infer the underlying implement of each node.

While the frontend only distinguish between Load/Store/Compute Node,
each of these nodes can have different underlying implementation based
on their layout. For instance, a LoadNode can be AuxLoad, Row/Col/Scalar broadcast, etc.
This pass infers the underlying impl of each node
"""
import cutlass.backend.evt.backend as evt_backend
from cutlass.backend.evt.ir import DAGIR, LoadNode
from cutlass.backend.evt.passes.pass_fix_element_d import PassFixElementD
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_no_op_elimination import PassNoOpElimination
from cutlass.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation
from cutlass.backend.evt.passes.util import cc_map

class PassGetImpl(EVTPassBase):
    """
    While the frontend only distinguish between Load/Store/Compute Node,
    each of these nodes can have different underlying implementation based
    on their layout. For instance, a LoadNode can be AuxLoad, Row/Col/Scalar broadcast, etc.
    This pass infers the underlying impl of each node
    """
    dependencies = [PassShapeTypePropagation, PassFixElementD]

    def __init__(self, dag_ir: DAGIR) -> None:
        super().__init__(dag_ir)
        self.no_op_elimination = PassNoOpElimination(dag_ir)

    def requires(self) -> None:
        if not self.dag_ir.has_node('accum'):
            raise SyntaxError("Cannot find 'accum' in the argument list.")

    def call(self):
        accumulator: LoadNode = self.dag_ir.get_node_meta('accum')
        problem_size = accumulator.tensor.shape
        for node_meta in self.dag_ir.node_metas_topological_order():
            node_meta.get_underlying_impl(problem_size)

    def ensures(self) -> None:
        self.no_op_elimination()
        for node_meta in self.dag_ir.nodes_meta:
            node_impl_ccs = getattr(evt_backend, f'sm{cc_map[self.cc]}_nodes')
            node_meta.underlying_impl = getattr(node_impl_ccs, f'Sm{cc_map[self.cc]}' + node_meta.underlying_impl.__class__.__name__)(node_meta)