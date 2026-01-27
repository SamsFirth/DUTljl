"""
Construct the epilogue visitor argument type
"""
from cutlass.backend.c_types import visitor_factory
from cutlass.backend.evt.ir import TopoVisitorNode
from cutlass.backend.evt.passes.pass_dag_2_tree import PassDAG2Tree
from cutlass.backend.evt.passes.pass_get_impl import PassGetImpl
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation

class PassGetArgumentType(EVTPassBase):
    """
    Construct the epilogue visitor argument type
    """
    dependencies = [PassShapeTypePropagation, PassDAG2Tree, PassGetImpl]

    def requires(self) -> None:
        if self.cc == 90 and (not self.dag_ir.has_node('D')):
            raise SyntaxError("Sm90 EVT requires the epilogue to have a returned tensor D, but the variable 'D' is not found in the return values.")

    def call(self):
        nodes = self.dag_ir.nodes_topological_order()
        self.argument_types = {}
        for node in nodes:
            meta = self.dag_ir.get_node_meta(node)
            if not meta.disabled:
                self.argument_types[node] = meta.underlying_impl.argument_type
            if node == 'D' and self.cc == 90:
                continue
            if isinstance(meta, TopoVisitorNode):
                self.get_dag_argument_type(node)
            else:
                self.get_evt_argument_type(node)
        self.cc_specific_method(self.set_argument_type)()

    def get_evt_argument_type(self, node):
        input_types = [self.argument_types[child] for child in self.dag_ir.get_all_inputs(node)]
        if len(input_types) > 0:
            self.argument_types[node] = visitor_factory(input_types + [self.argument_types[node]], self.dag_ir.get_all_inputs(node) + [node])

    def get_dag_argument_type(self, node):
        meta = self.dag_ir.get_node_meta(node)
        subgraph = meta.subgraph
        subgraph_nodes = subgraph.nodes_topological_order()
        for n in subgraph_nodes:
            m = subgraph.get_node_meta(n)
            if m.disabled:
                continue
            else:
                self.argument_types[n] = m.underlying_impl.argument_type
        input_types = [self.argument_types[child] for child in subgraph_nodes[:-1]]
        if len(input_types) > 0:
            self.argument_types[node] = visitor_factory(input_types, subgraph_nodes[:-1])

    def set_argument_type(self):
        pass

    def sm90_set_argument_type(self):
        self.dag_ir.epilogue_thread_type = self.argument_types[self.dag_ir.get_all_inputs('D')[0]]
        self.dag_ir.arg_d_type = self.dag_ir.get_node_meta('D').underlying_impl.argument_type_d
        if self.dag_ir.has_node('C'):
            self.dag_ir.arg_c_type = self.dag_ir.get_node_meta('C').underlying_impl.argument_type_c
        else:
            self.dag_ir.arg_c_type = self.dag_ir.arg_d_type

    def sm80_set_argument_type(self):
        nodes = self.dag_ir.nodes_topological_order()
        self.dag_ir.epilogue_thread_type = self.argument_types[nodes[-1]]