"""
Compute the shared memory size in bytes
"""
import cutlass_library
from pycute import shape_div, product
import cutlass
from cutlass.backend.evt.ir import TopoVisitorNode, DAGIR
from cutlass.backend.library import DataTypeSize

class GetSmemSize:
    """
    Get the size in byte of shared memory used by the kernel
    """

    def __init__(self, dag_ir: DAGIR) -> None:
        self.dag_ir = dag_ir
        self.cc = self.dag_ir.cc

    def sm90_epilogue_tile(self, tile_description):
        schedule = tile_description.epilogue_schedule
        if schedule == cutlass_library.EpilogueScheduleType.TmaWarpSpecialized:
            epilogue_tile_mn = (64, 32)
        elif schedule == cutlass_library.EpilogueScheduleType.TmaWarpSpecializedCooperative:
            if tile_description.threadblock_shape[0] >= 128:
                epilogue_tile_mn = (128, 32)
            else:
                epilogue_tile_mn = (64, 32)
        else:
            raise NotImplementedError(f'Unsupported schedule: {schedule}')
        stages_d = 2
        epi_tiles = product(shape_div(tuple(tile_description.threadblock_shape)[:2], epilogue_tile_mn))
        if self.dag_ir.has_node('C'):
            element_c = self.dag_ir.get_node_meta('C').element
        else:
            element_c = None
        element_d = self.dag_ir.get_node_meta('D').element
        if element_c == element_d:
            reuse_smem_c = True
        else:
            reuse_smem_c = False
        stages_c = max(epi_tiles, stages_d + 1) if reuse_smem_c else epi_tiles
        self.cta_tile_mnk = tuple(tile_description.threadblock_shape)
        self.epilogue_tile_mn = epilogue_tile_mn
        self.epi_tiles = epi_tiles
        self.stages_c = stages_c
        self.stages_d = stages_d
        self.reuse_smem_c = reuse_smem_c
        self.element_c = element_c
        self.element_d = element_d
        self.is_source_supported = element_c is not None

    def sm90_epilogue_smem_size(self, tile_description):
        """
        Compute the shared memory size of sm90 collective epilogue
        """
        self.sm90_epilogue_tile(tile_description)
        nodes = self.dag_ir.nodes_topological_order()
        self.smem_types = {}
        for node in nodes:
            meta = self.dag_ir.get_node_meta(node)
            if not meta.disabled:
                self.smem_types[node] = meta.underlying_impl.get_smem_size(self.cta_tile_mnk, self.epilogue_tile_mn, self.stages_c, self.stages_d, self.epi_tiles)
            if node == 'D':
                continue
            if isinstance(meta, TopoVisitorNode):
                self.get_dag_smem_type(node)
            else:
                self.get_evt_smem_type(node)
        thread_smem_size = self.smem_types[self.dag_ir.get_all_inputs('D')[0]][0]
        tensors = []
        if self.is_source_supported:
            smem_C = DataTypeSize[self.element_c] * product(self.epilogue_tile_mn) * self.stages_c // 8
            tensors.append((smem_C, 128))
        else:
            tensors.append((0, 1))
        if self.reuse_smem_c:
            tensors.append((0, 128))
        else:
            smem_D = DataTypeSize[self.element_d] * product(self.epilogue_tile_mn) * self.stages_d // 8
            tensors.append((smem_D, 128))
        tensors.append((thread_smem_size, 128))
        tensor_smem_size = self.get_struct_size(tensors)
        pipeline_smem_size = (8 * self.stages_c * 2, 8)
        smem_size = self.get_struct_size([tensor_smem_size, pipeline_smem_size])
        return smem_size[0]

    def __call__(self, tile_description):
        return getattr(self, f'sm{self.cc}_epilogue_smem_size')(tile_description)

    @staticmethod
    def get_visitor_size(members: list, ebo: bool):
        """
        Get the size of struct in bytes
        """
        offset = 0
        max_alignment = 1
        if len(members) > 0:
            for _, alignment in members:
                max_alignment = max(max_alignment, alignment)
            for type_size, _ in members:
                if type_size != 0:
                    offset = (offset + max_alignment - 1) // max_alignment * max_alignment
                if type_size == 0 and (not ebo):
                    offset += 1
                else:
                    offset += type_size
            offset = (offset + max_alignment - 1) // max_alignment * max_alignment
            return (offset, max_alignment)
        else:
            return (1, 1)

    def get_struct_size(self, members: list):
        """
        Get the size of struct in bytes
        """
        return self.get_visitor_size(members, False)

    def get_evt_smem_type(self, node):
        input_types = [self.smem_types[child] for child in self.dag_ir.get_all_inputs(node)]
        input_types.append(self.smem_types[node])
        if len(input_types) > 1:
            ebo = len(input_types) > 4
            self.smem_types[node] = self.get_visitor_size(input_types, ebo)

    def get_dag_smem_type(self, node):
        meta = self.dag_ir.get_node_meta(node)
        subgraph = meta.subgraph
        subgraph_nodes = subgraph.nodes_topological_order()
        for n in subgraph_nodes:
            m = subgraph.get_node_meta(n)
            if m.disabled:
                continue
            else:
                self.smem_types[n] = m.underlying_impl.get_smem_size(self.cta_tile_mnk, self.epilogue_tile_mn, self.stages_c, self.stages_d, self.epi_tiles)
        input_types = [self.smem_types[child] for child in subgraph_nodes[:-1]]
        if len(input_types) > 0:
            ebo = len(input_types) > 4
            self.smem_types[node] = self.get_visitor_size(input_types, ebo)