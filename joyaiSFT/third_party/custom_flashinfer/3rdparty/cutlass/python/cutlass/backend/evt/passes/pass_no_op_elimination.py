"""
No op elimination node
"""
from typing import Any
from cutlass.backend.evt.ir import NoOpImpl
from cutlass.backend.evt.passes.pass_manager import EVTPassBase

class PassNoOpElimination(EVTPassBase):
    """
    The dead node elimination pass removes nodes with NoOpImpl in DAG IR
    """
    dependencies = []

    def call(self) -> Any:
        for node in self.dag_ir.nodes_topological_order():
            node_meta = self.dag_ir.get_node_meta(node)
            if isinstance(node_meta.underlying_impl, NoOpImpl):
                self.dag_ir.replace_all_uses_with(node, self.dag_ir.get_all_inputs(node)[0])