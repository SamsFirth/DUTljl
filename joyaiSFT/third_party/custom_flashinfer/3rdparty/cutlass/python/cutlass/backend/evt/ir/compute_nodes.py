"""
Python registration for compute nodes in EVT
"""
from cutlass.backend.evt.ir.node import NodeBase, ImplBase
from cutlass.backend.library import FloatRoundStyle

class ComputeImplBase(ImplBase):
    """
    Base class for compute implementation
    """

    def __init__(self, node) -> None:
        super().__init__(node)

class ComputeImpl(ComputeImplBase):
    """
    Implementation for Compute Node
    """

    def __init__(self, node) -> None:
        super().__init__(node)
        self.fn = node.fn
        self.element_output = node.element_output
        self.element_compute = node.element_compute
        self.round_style = node.round_style

    @staticmethod
    def match(node, problem_size: tuple):
        return True

class ComputeNode(NodeBase):
    """
    Compute Node in DAG IR
    """
    possible_impls = [ComputeImpl]

    def __init__(self, name: str, fn, element_output, element_compute, round_style=FloatRoundStyle.ToNearest) -> None:
        super().__init__(name)
        self.op = 'compute'
        self.fn = fn
        self.element_compute = element_compute
        self.round_style = round_style

    def type_propagation(self, *args, **kwargs):
        """
        Load node loads tensor under type `tensor.element` and returns an array of type `tensor.element`.
        """
        self.element = self.element_compute
        if not hasattr(self, 'element_output'):
            self.element_output = self.element