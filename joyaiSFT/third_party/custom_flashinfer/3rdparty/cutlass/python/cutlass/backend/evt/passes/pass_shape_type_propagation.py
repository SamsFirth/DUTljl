"""
Shape and type propagation pass
"""
from cutlass.backend.evt.ir.node import NodeBase
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_preprocess_red import PassPreprocessRed

class PassShapeTypePropagation(EVTPassBase):
    """
    Propagate the shape and type of all nodes
    """
    dependencies = [PassPreprocessRed]

    def call(self):
        for node in self.dag_ir.nodes_topological_order():
            node_meta: NodeBase = self.dag_ir.get_node_meta(node)
            input_node_metas = self.dag_ir.get_all_inputs_meta(node)
            node_meta.type_propagation(input_node_metas)
            node_meta.shape_propagation(input_node_metas)
        for node in reversed(self.dag_ir.nodes_topological_order()):
            node_meta: NodeBase = self.dag_ir.get_node_meta(node)
            input_node_metas = self.dag_ir.get_all_inputs_meta(node)
            node_meta.broadcast_propagation(input_node_metas)