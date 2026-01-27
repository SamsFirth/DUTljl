"""
Fix the element_output of producer of D.

In Sm90 epilogue visitor, the node writing D to gmem does not have internal
element converter, so the compute node producing D must have element_output = type(D).
"""
from cutlass.backend.evt.passes.pass_layout_elimination import PassLayoutManipulateElimination
from cutlass.backend.evt.passes.pass_manager import EVTPassBase

class PassFixElementD(EVTPassBase):
    """
    In Sm90 epilogue visitor, the node writing D to gmem does not have internal
    element converter, so the compute node producing D must have
    element_output = type(D)
    """
    dependencies = [PassLayoutManipulateElimination]

    def get_producer(self, node, element_D):
        node_meta = self.dag_ir.get_node_meta(node)
        if node_meta.op == 'compute':
            node_meta.element_output = element_D
        elif node_meta.op == 'store':
            self.get_producer(self.dag_ir.get_all_inputs(node)[0], element_D)

    def call(self):
        if self.dag_ir.has_node('D'):
            node_d_meta = self.dag_ir.get_node_meta('D')
            element_D = node_d_meta.store_tensor.element
            self.get_producer('D', element_D)