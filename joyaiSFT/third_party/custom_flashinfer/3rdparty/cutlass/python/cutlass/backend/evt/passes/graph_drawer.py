import subprocess
from cutlass_library import DataTypeTag
import pydot
from cutlass.backend.evt.ir.dag_ir import DAGIR
_COLOR_MAP = {'load': '"AliceBlue"', 'compute': 'LemonChiffon1', 'accumulator': 'LightGrey', 'store': 'PowderBlue', 'layout': 'lightseagreen', 'dag': 'darkorange'}

class EVTGraphDrawer:
    """
    Visualize a EVT DAGIR with graphviz
    """

    def __init__(self, graph: DAGIR, name: str):
        self._name = name
        self._dot_graphs = {}
        self._dot_graphs[name] = self._to_dot(graph, name)

    def _get_node_style(self, node):
        template = {'shape': 'record', 'fillcolor': '#CAFFE3', 'style': '"filled,rounded"', 'fontcolor': '#000000'}
        if node.op in _COLOR_MAP:
            template['fillcolor'] = _COLOR_MAP[node.op]
        else:
            raise NotImplementedError('unknown node op')
        if node.disabled:
            template['fontcolor'] = 'grey'
            template['fillcolor'] = 'white'
        return template

    def _get_node_label(self, node):
        label = '{' + f'name={node.name}|op={node.op}'
        if node.op == 'layout':
            label += f'|fn={node.fn.__name__}'
            for key in node.kwargs:
                label += f'|{key}={node.kwargs[key]}'
        if node.underlying_impl is not None:
            label += f'|impl={type(node.underlying_impl).__name__}'
            if node.op == 'load':
                label += f'|element_output={DataTypeTag[node.underlying_impl.element]}'
            elif node.op == 'compute':
                label += f'|element_compute={DataTypeTag[node.underlying_impl.element_compute]}|element_output={DataTypeTag[node.underlying_impl.element_output]}'
            elif node.op == 'store':
                label += f'|element_store={DataTypeTag[node.underlying_impl.element]}|element_output={DataTypeTag[node.underlying_impl.element_output]}'
            elif node.op == 'dag':
                label += f'|element_output={DataTypeTag[node.underlying_impl.element_output]}'
        if node.tensor is not None:
            shape = node.tensor.shape
            stride = node.tensor.stride
            label += f'|shape={shape}|stride={stride}'
        if hasattr(node, 'store_tensor'):
            if node.store_tensor is not None:
                store_shape = node.store_tensor.shape
                store_stride = node.store_tensor.stride
                label += f'|store_shape={store_shape}|stride_stride={store_stride}'
        label += '}'
        return label

    def _to_dot(self, graph: DAGIR, name: str):
        dot_graph = pydot.Dot(name, randir='TB')
        for node in graph.nodes_meta:
            style = self._get_node_style(node)
            label = self._get_node_label(node)
            dot_node = pydot.Node(node.name, label=label, **style)
            dot_graph.add_node(dot_node)
            if node.op == 'dag':
                dot_subgraph = self._to_dot(node.subgraph, name=node.name)
                self._dot_graphs[node.name] = dot_subgraph
        for src, dst in graph.edges:
            weight = graph.get_edge_weight(src, dst)
            dot_graph.add_edge(pydot.Edge(src, dst, label=weight))
        return dot_graph

    def get_dot_graph(self) -> pydot.Dot:
        return [(key, self.get_dot_graph_by_name(key)) for key in self._dot_graphs.keys()]

    def get_dot_graph_by_name(self, name) -> pydot.Dot:
        return self._dot_graphs[name]

    def get_main_dot_graph(self) -> pydot.Dot:
        return self._dot_graphs[self._name]