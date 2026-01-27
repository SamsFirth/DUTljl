"""
Base class for Python EVT Frontend
"""
from typing import Union
from cutlass_library import DataType
from cutlass.backend.evt.ir import ComputeNode, DAGIR, LayoutNode, LoadNode, StoreNode
from cutlass.backend.evt.passes import EVTGraphDrawer, EVTPassManager, GetSmemSize, PassDAG2Tree, PassGetArgumentType, PassGetImpl, PassFixElementD, PassLayoutManipulateElimination, PassPreprocessRed, PassShapeTypePropagation
from cutlass.backend.utils import device_cc
from cutlass.epilogue.evt_ops import permute, reshape
from cutlass.utils.datatypes import library_type

class EVTFrontendBase:
    layout_fns = {'permute': permute, 'reshape': reshape}

    def __init__(self, element_compute=DataType.f32, cc=None, additional_passes=[], **kwargs) -> None:
        self.cc = cc if cc else device_cc()
        self.element_compute = library_type(element_compute)
        self.dag_ir = DAGIR(self.element_compute, self.cc)
        self.compute_cnt = 0
        self.layout_cnt = 0
        self.pass_manager = EVTPassManager(self.dag_ir, [PassPreprocessRed, PassGetArgumentType, PassShapeTypePropagation, PassLayoutManipulateElimination, PassGetImpl, PassDAG2Tree, PassFixElementD] + additional_passes)
        if self.cc == 80:
            self._epilogue_stages = 1
        else:
            self._epilogue_stages = None

    @property
    def epilogue_stages(self):
        return self._epilogue_stages

    @epilogue_stages.setter
    def epilogue_stages(self, stages):
        self._epilogue_stages = stages

    def parse(self, *args, **kwargs):
        raise NotImplementedError(f"The 'parse' function must be overloaded in frontend class")

    def trace(self, *args, **kwargs):
        self.parse(*args, **kwargs)
        self.pass_manager()
        self.epilogue_thread_type = self.dag_ir.epilogue_thread_type
        if self.cc == 90:
            self.arg_c_type = self.dag_ir.arg_c_type
            self.arg_d_type = self.dag_ir.arg_d_type
        self.reduction_names = self.dag_ir.reduction_names

    def add_node(self, node):
        self.dag_ir.add_node(node)

    def add_edge(self, src, tgt, weight=0):
        self.dag_ir.add_edge(src, tgt, weight=weight)

    def set_tensor(self, node_name, example):
        """
        Add an example tensor to node {node_name} in the DAG IR
        """
        meta = self.dag_ir.get_node_meta(node_name)
        meta.tensor = {'tensor': example}

    def set_store_tensor(self, node_name, example):
        """
        Add an example tensor to node {node_name} in the DAG IR
        """
        meta = self.dag_ir.get_node_meta(node_name)
        meta.store_tensor = {'tensor': example}

    def mark_output(self, node_name):
        """
        Mark a store node as output
        """
        meta = self.dag_ir.get_node_meta(node_name)
        if not isinstance(meta, StoreNode):
            raise ValueError(f'Only StoreNodes can be marked as output. Got {type(meta).__name__}: {node_name}')
        meta.is_output = True

    def add_load_node(self, name, example):
        """
        Add a Load node to DAG IR
        :param name: name of the loaded variable
        :type name: str
        :param example: example input
        :type example: np.ndarray|torch.Tensor|cupy.ndarray|float
        """
        if name is None:
            raise ValueError(f'Name is not provided.')
        if example is None:
            raise ValueError(f'Example input for {name} is not provided.')
        load_node = LoadNode(name)
        load_node.tensor = {'tensor': example}
        if name == 'accum':
            if load_node.tensor.rank == 2:
                new_shape = tuple([1] + list(load_node.tensor.shape))
                load_node.tensor.broadcast(new_shape)
            elif load_node.tensor.rank < 2 or load_node.tensor.rank > 3:
                raise ValueError(f"Expect example inputs for 'accum' be a rank-2 or rank-3 tensor. Got {load_node.tensor.shape}.")
        self.add_node(load_node)

    def add_imm(self, value: Union[float, int]):
        """
        Add an immediate scalar value to DAG IR
        :param value: the value of the immediate scalar
        :type value: float
        """
        try:
            value = float(value)
        except:
            raise ValueError(f'{type(value).__name__} cannot be converted to float.')
        name = f'imm_{value}'.replace('.', '_')
        load_node = LoadNode(name)
        load_node.tensor = {'tensor': value, 'is_constant': True}
        self.add_node(load_node)
        return name

    def add_compute_node(self, op, name=None):
        """
        Add a compute node.
        :param op: the computation op
        :param name: the node name (optional)
        :type name: str
        :return: the name of the compute node
        """
        if name is None:
            name = f'compute_{self.compute_cnt}'
            self.compute_cnt += 1
        compute_node = ComputeNode(name=name, fn=op, element_output=self.element_compute, element_compute=self.element_compute)
        self.add_node(compute_node)
        return compute_node.name

    def add_layout_node(self, op, kwargs, name=None):
        """
        Add a layout node.
        :param op: the layout op
        :type op: evt_ops
        :param name: the node name (optional)
        :type name: str
        :return: the name of the layout node
        """
        if name is None:
            name = f'layout_{self.layout_cnt}'
            self.layout_cnt += 1
        layout_node = LayoutNode(name=name, fn=op, kwargs=kwargs)
        self.add_node(layout_node)
        return layout_node.name

    def add_store_node(self, name):
        store_node = StoreNode(name)
        self.add_node(store_node)

    def visualize(self, name='dag_ir'):
        """
        Visualize the dag ir with svg file
        :param name: the name of the graph
        """
        drawer = EVTGraphDrawer(self.dag_ir, name)
        try:
            for name, graph in drawer.get_dot_graph():
                graph.write_svg(f'./{name}.svg')
        except:
            raise RuntimeError("'dot' is not found in path. GraphDrawer is disabled. Please install it with 'sudo apt-get install graphviz'.")

    def get_smem_size(self, tile_description):
        """
        Get the shared memory size of the epilogue
        """
        smem_size = GetSmemSize(self.dag_ir)(tile_description)
        return smem_size