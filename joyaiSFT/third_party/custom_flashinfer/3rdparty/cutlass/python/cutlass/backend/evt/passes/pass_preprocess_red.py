"""
Preprocess the reduction nodes.

The parser treats reduction as Compute(op=(reg_reduce_fn, gmem_reduce_fn)) - Store()
This pass fuses these into a single store node, and then replaces all uses of the
current node with the new store node.
"""
from cutlass.backend.evt.ir import ComputeNode, StoreNode
from cutlass.backend.evt.passes.pass_manager import EVTPassBase

class PassPreprocessRed(EVTPassBase):
    """
    Preprocess red nodes
    """

    def call(self):
        red_compute_nodes = []
        for node_meta in self.dag_ir.nodes_meta:
            if isinstance(node_meta, ComputeNode):
                if type(node_meta.fn) == tuple:
                    red_compute_nodes.append(node_meta.name)
        for node in red_compute_nodes:
            users = self.dag_ir.get_users(node)
            inputs = self.dag_ir.get_all_inputs(node)
            assert len(users) == 1
            assert len(inputs) == 1
            user = users[0]
            input = inputs[0]
            user_meta = self.dag_ir.get_node_meta(user)
            assert isinstance(user_meta, StoreNode)
            assert self.dag_ir.out_degree(user) == 0
            node_meta = self.dag_ir.get_node_meta(node)
            user_meta.reg_reduce_fn, user_meta.gmem_reduce_fn = node_meta.fn
            user_meta.element_compute = node_meta.element_compute
            user_meta.round_style = node_meta.round_style
            self.dag_ir.remove_edge(input, node)
            input_users = self.dag_ir.get_users(input)
            for iu in input_users:
                weight = self.dag_ir.get_edge_weight(input, iu)
                self.dag_ir.add_edge(user, iu, weight)
                self.dag_ir.remove_edge(input, iu)
            self.dag_ir.add_edge(input, user)
            self.dag_ir.remove_node(node)
            self.dag_ir.reduction_names.append(user)