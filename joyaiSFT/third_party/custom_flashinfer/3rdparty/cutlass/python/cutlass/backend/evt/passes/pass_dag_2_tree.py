"""
Merge non-tree sub-graphs of the DAG IR into a single DAG. The fused DAG will be implemented
by the topological visitor, while the rest of the graph will be implemented with the tree visitor.
"""
from copy import deepcopy
from cutlass.backend.evt.ir import DAGIR, TopoVisitorNode
from cutlass.backend.evt.passes.pass_get_impl import PassGetImpl
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation

class PassDAG2Tree(EVTPassBase):
    """
    Convert the DAG IR to Tree by fusing subgraphs
    """
    dependencies = [PassShapeTypePropagation, PassGetImpl]

    def call(self):
        multi_parent_nodes = []
        for node in self.dag_ir.nodes_topological_order():
            if self.dag_ir.out_degree(node) > 1:
                multi_parent_nodes.append(node)
        for node in multi_parent_nodes:
            if not self.dag_ir.has_node(node):
                continue
            if self.dag_ir.out_degree(node) <= 1:
                continue
            reachable_nodes = []
            for parent in self.dag_ir.get_users(node):
                reachable_nodes.append(set(self.dag_ir.all_reachable_nodes(parent)))
            common_items = set.intersection(*reachable_nodes)
            if len(common_items) > 0:
                topo_order = self.dag_ir.nodes_topological_order()
                lca = None
                topo_idx = -1
                for item in common_items:
                    if lca is None:
                        lca = item
                        topo_idx = topo_order.index(item)
                    elif topo_idx > topo_order.index(item):
                        lca = item
                        topo_idx = topo_order.index(item)
                node_to_fuse = set.union(*reachable_nodes).difference(common_items)
                node_to_fuse.add(lca)
                all_input_nodes = []
                all_output_nodes = []
                for node in node_to_fuse:
                    all_input_nodes.append(set(self.dag_ir.get_all_inputs(node)))
                    all_output_nodes.append(set(self.dag_ir.get_users(node)))
                all_input_nodes = set.union(*all_input_nodes)
                all_output_nodes = set.union(*all_output_nodes)
                new_subgraph_nodes = set.union(node_to_fuse, all_input_nodes, all_output_nodes)
                subgraph_ = self.dag_ir._graph.subgraph(new_subgraph_nodes)
                subgraph = DAGIR()
                for node in subgraph_.nodes:
                    meta = deepcopy(self.dag_ir.get_node_meta(node))
                    if node not in node_to_fuse:
                        meta.disabled = True
                    subgraph.add_node(meta)
                for edge in subgraph_.edges:
                    subgraph.add_edge(edge[0], edge[1], self.dag_ir.get_edge_weight(edge[0], edge[1]))
                dag_node = TopoVisitorNode(name=f'dag_{lca}', subgraph=subgraph, output_node=self.dag_ir.get_node_meta(lca))
                self.dag_ir.add_node(dag_node)
                for idx, node in enumerate(all_input_nodes):
                    self.dag_ir.add_edge(node, dag_node.name, weight=idx)
                self.dag_ir.replace_all_uses_with(lca, dag_node.name)
                node_to_fuse.remove(lca)
                for node in node_to_fuse:
                    self.dag_ir.remove_node(node)
            else:
                raise NotImplementedError('No LCA found. Consider SplitTreeVisitor.')

    def ensures(self) -> None:
        for node in self.dag_ir.nodes:
            out_degree = self.dag_ir.out_degree(node)
            if out_degree > 1:
                raise RuntimeError(f'PassDAG2Tree failed. Node {node} still have outdegree = {out_degree}')