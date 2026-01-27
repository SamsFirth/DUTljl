"""
Pass manager for DAG IR.
"""
from typing import Any
import networkx as nx
from cutlass.backend.evt.ir import DAGIR
from cutlass.backend.evt.passes.util import cc_map

class EVTPassBase:
    """
    Base class for EVT Passes
    """
    dependencies = []

    def __init__(self, dag_ir: DAGIR) -> None:
        self.dag_ir = dag_ir
        self.cc = self.dag_ir.cc

    def requires(self) -> None:
        """
        This function will be called before the pass is run.
        """
        pass

    def call(self) -> None:
        """
        The pass that is run through the self.dag_ir
        """
        raise NotImplementedError(f'__call__ is not overwritten in Pass {self.__class__.__name__}')

    def ensures(self) -> None:
        """
        This function will be called after the pass is run.
        """
        pass

    def __call__(self) -> Any:
        self.requires()
        self.call()
        self.ensures()

    def cc_specific_method(self, func):
        """
        This enables defining function that behaves differently under different cc
        The simplest example of using this function is the following

        .. highlight:: python
        .. code-block:: python

        class ExamplePass(EVTPassBase):

            def call(sekf):
                # This automatically select the smXX_func based on current cc
                self.cc_specific_method(self.func)()

            # Interface func, can be empty
            def func(self):
                pass

            # Sm90 specific func
            def sm90_func(self):
                // sm90 specific method
                return

            # Sm80 specific func
            def sm80_func(self):
                // sm80 specific method
                return
        """
        func_name = f'sm{cc_map[self.cc]}_{func.__name__}'
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise NotImplementedError(f'func {func.__name__} is not overwritten for Sm{self.cc}')

class EVTPassManager(nx.DiGraph):
    """
    Topological-based Pass Manager.
    Each registered pass has a list of dependencies. The pass manager organizes
    the passes as a DAG and launch the compiler passes under topological order.
    """

    def __init__(self, dag_ir: DAGIR, pass_list):
        super().__init__()
        self.dag_ir = dag_ir
        for pass_cls in pass_list:
            self.add_pass(pass_cls)
        self.sorted_passes = self.schedule()

    def get_callable(self, pass_name):
        """
        Return the callable of the pass
        """
        return self.nodes[pass_name]['callable']

    def add_pass(self, pass_cls):
        """
        Add a pass to the pass manager
        :param pass_cls: the class of pass
        :type pass_cls: derived class of EVTPassBase
        """
        name = pass_cls.__name__
        pass_callable = pass_cls(self.dag_ir)
        self.add_node(name, callable=pass_callable)

    def schedule(self):
        """
        Schedule the added passes under topological order
        """
        for pass_name in self.nodes:
            callable = self.get_callable(pass_name)
            for dependency_cls in callable.dependencies:
                self.add_edge(dependency_cls.__name__, type(callable).__name__)
        return list(nx.topological_sort(self))

    def __call__(self) -> Any:
        """
        Launch the registered passes
        """
        for pass_name in self.sorted_passes:
            callable = self.get_callable(pass_name)
            callable()