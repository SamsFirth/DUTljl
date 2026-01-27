"""
Eliminate layout manipulation nodes
"""
from copy import deepcopy
from cutlass.backend.evt.ir import DAGIR, LayoutNode
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation

class PassLayoutManipulateElimination(EVTPassBase):
    """
    Eliminate layout manipulation nodes
    """
    dependencies = [PassShapeTypePropagation]

    def __init__(self, dag_ir: DAGIR) -> None:
        super().__init__(dag_ir)
        self.copy_cnt = 0

    def call(self):
        self.layout_nodes_worklist = self.get_all_layout_nodes()
        while len(self.layout_nodes_worklist) > 0:
            node = self.layout_nodes_worklist.pop(0)
            direction = self.get_propagation_direction(node)
            self.visited = []
            getattr(self, f'propagate_to_{direction}')(self.dag_ir.get_node_meta(node), node)
            input_node = self.dag_ir.get_all_inputs(node)[0]
            self.dag_ir.replace_all_uses_with(node, input_node)

    def get_all_layout_nodes(self):
        layout_nodes = []
        for node_meta in reversed(self.dag_ir.node_metas_topological_order()):
            if isinstance(node_meta, LayoutNode):
                layout_nodes.append(node_meta.name)
        return layout_nodes

    def get_propagation_direction(self, node: str):
        """
        The logic is propagating all layout nodes away from the accumulator node.
        """
        self.visited = []
        self.get_influenced_users(node)
        nodes_influenced_dir_users = self.visited
        self.visited = []
        self.get_influenced_inputs(node)
        nodes_influenced_dir_inputs = self.visited
        if 'accum' in nodes_influenced_dir_users and 'accum' not in nodes_influenced_dir_inputs:
            return 'inputs'
        elif 'accum' not in nodes_influenced_dir_users and 'accum' in nodes_influenced_dir_inputs:
            return 'users'
        else:
            raise RuntimeError('Unsolved propagation direction')

    def get_influenced_users(self, node: str):
        if node in self.visited:
            return
        self.visited.append(node)
        users = self.dag_ir.get_users(node)
        for user in users:
            self.get_influenced_users(user)
        user_inputs = []
        for user in users:
            user_inputs.append(set(self.dag_ir.get_all_inputs(user)))
        if len(user_inputs) > 0:
            user_inputs = set.union(*user_inputs)
            user_inputs.remove(node)
            for input in user_inputs:
                self.get_influenced_inputs(input)

    def get_influenced_inputs(self, node: str):
        if node in self.visited:
            return
        self.visited.append(node)
        inputs = self.dag_ir.get_all_inputs(node)
        for input in inputs:
            self.get_influenced_inputs(input)
        input_users = []
        for input in inputs:
            input_users.append(set(self.dag_ir.get_users(input)))
        if len(input_users) > 0:
            input_users = set.union(*input_users)
            input_users.remove(node)
            for user in input_users:
                self.get_influenced_users(user)

    def add_copy_before(self, layout_node_meta: LayoutNode, target: str):
        copied_node_meta = deepcopy(layout_node_meta)
        copied_node = f'{copied_node_meta.name}_copy{self.copy_cnt}'
        self.copy_cnt += 1
        copied_node_meta.name = copied_node
        self.dag_ir.add_node(copied_node_meta)
        target_inputs = self.dag_ir.get_all_inputs(target)
        for src in target_inputs:
            self.dag_ir.remove_edge(src, target)
            self.dag_ir.add_edge(src, copied_node)
        self.dag_ir.add_edge(copied_node, target)
        self.layout_nodes_worklist.append(copied_node)

    def add_copy_after(self, layout_node_meta: LayoutNode, target: str):
        copied_node_meta = deepcopy(layout_node_meta)
        copied_node = f'{copied_node_meta.name}_copy{self.copy_cnt}'
        self.copy_cnt += 1
        copied_node_meta.name = copied_node
        self.dag_ir.add_node(copied_node_meta)
        users = self.dag_ir.get_users(target)
        for user in users:
            self.dag_ir.remove_edge(target, user)
            self.dag_ir.add_edge(copied_node, user)
        self.dag_ir.add_edge(target, copied_node)
        self.layout_nodes_worklist.append(copied_node)

    def propagate_to_users(self, layout_node_meta: LayoutNode, node: str):
        """
        Propagate layout node to users
        """
        if node in self.visited:
            return
        self.visited.append(node)
        node_meta = self.dag_ir.get_node_meta(node)
        if layout_node_meta.name != node:
            if isinstance(node_meta, LayoutNode):
                self.add_copy_before(layout_node_meta, node)
                return
            else:
                layout_node_meta.apply_to_user(node_meta)
        users = self.dag_ir.get_users(node)
        user_inputs = []
        for user in users:
            user_inputs.append(set(self.dag_ir.get_all_inputs(user)))
        for user in users:
            self.propagate_to_users(layout_node_meta, user)
        if len(user_inputs) > 0:
            user_inputs = set.union(*user_inputs)
            user_inputs.remove(node)
            for input in user_inputs:
                self.propagate_to_inputs(layout_node_meta.get_inverse_node(), input)

    def propagate_to_inputs(self, layout_node_meta: LayoutNode, node: str):
        """
        Propagate layout node to inputs
        """
        if node in self.visited:
            return
        self.visited.append(node)
        node_meta = self.dag_ir.get_node_meta(node)
        if layout_node_meta.name != node:
            if isinstance(node_meta, LayoutNode):
                self.add_copy_after(layout_node_meta, node)
                return
            else:
                layout_node_meta.apply_to_input(node_meta)
        inputs = self.dag_ir.get_all_inputs(node)
        input_users = []
        for input in inputs:
            input_users.append(set(self.dag_ir.get_users(input)))
        for input in inputs:
            self.propagate_to_inputs(layout_node_meta, input)
        if len(input_users) > 0:
            input_users = set.union(*input_users)
            input_users.remove(node)
            for user in input_users:
                self.propagate_to_users(layout_node_meta.get_inverse_node(), user)