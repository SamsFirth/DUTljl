"""
Utilities for emitting CUTLASS >= 3 convolution kernels
"""
import enum
import os.path
import shutil
import logging
from string import Template
try:
    import builtins
    if hasattr(builtins, 'CUTLASS_IGNORE_PACKAGE') and CUTLASS_IGNORE_PACKAGE == True:
        raise ImportError('Disabling attempt to import cutlass_library')
    from cutlass_library.library import *
except ImportError:
    from library import *
_LOGGER = logging.getLogger(__name__)

class EmitConv3xInstance:

    def __init__(self):
        _LOGGER.debug('*** EmitConv3xInstance::__init__')
        self.template = '\n\n// CUTLASS >= 3 convolution ${conv_kind_name} kernel instance "${operation_name}"\nusing ${operation_name}_epilogue =\n  typename cutlass::epilogue::collective::CollectiveBuilder<\n    ${arch},\n    ${opcode_class_epi},\n    ${output_cta_tile_shape},        // output cta tile shape\n    ${cluster_shape},                // cluster shape\n    ${epi_tile_mn},\n    ${element_accumulator},\n    ${element_compute},\n    ${element_c}, ${layout_c}, 128 / cute::sizeof_bits_v<${element_c}>,\n    ${element_d}, ${layout_d}, 128 / cute::sizeof_bits_v<${element_d}>,\n    ${epilogue_schedule}\n    // , class FusionOpOrCallbacks = cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute>\n  >::CollectiveOp;\n\nusing ${operation_name}_mainloop =\n  typename cutlass::conv::collective::CollectiveBuilder<\n    ${arch},\n    ${opcode_class_main},\n    ${conv_kind},         // kFprop, kDgrad, or kWgrad\n    ${element_a}, ${layout_a}, 128 / cute::sizeof_bits_v<${element_a}>,\n    ${element_b}, ${layout_b}, 128 / cute::sizeof_bits_v<${element_b}>,\n    ${element_accumulator},\n    ${mma_tile_shape},        // mma tile shape\n    ${cluster_shape},         // cluster shape\n    ${stages},\n    ${kernel_schedule}\n  >::CollectiveOp;\n\nusing ${operation_name}_problem_shape = cutlass::conv::ConvProblemShape<${conv_kind}, ${operation_name}_mainloop::NumSpatialDimensions>;\n\n// Unit tests call this "ConvKernel".\n// Conv operator ${operation_name}\nusing ${operation_name}_base = cutlass::conv::kernel::ConvUniversal<\n    ${operation_name}_problem_shape,\n    ${operation_name}_mainloop,\n    ${operation_name}_epilogue,\n    ${tile_scheduler}\n  >;\n'

    def arch_number_to_type(self, arch: int) -> str:
        return f'cutlass::arch::Sm{arch}'

    def output_cta_tile_shape(self, operation, cta_m, cta_n, cta_k) -> str:
        m_template = 'cute::_${cta_m}'
        if operation.conv_kind == ConvKind.Wgrad:
            n_template = 'cute::Shape<cute::_${cta_n}>'
        else:
            n_template = 'cute::_${cta_n}'
        k_template = 'cute::Shape<cute::_${cta_k}>'
        output_cta_tile_shape_template = f'cute::Shape<{m_template}, {n_template}, {k_template}>'
        values = {'cta_m': cta_m, 'cta_n': cta_n, 'cta_k': cta_k}
        return Template(output_cta_tile_shape_template).substitute(values)

    def mma_tile_shape(self, operation, cta_m, cta_n, cta_k) -> str:
        mma_m = cta_m
        mma_n = cta_n
        mma_k = cta_k
        m_template = 'cute::_${mma_m}'
        if operation.conv_kind == ConvKind.Wgrad:
            n_template = 'cute::Shape<cute::_${mma_n}>'
        else:
            n_template = 'cute::_${mma_n}'
        k_template = 'cute::Shape<cute::_${mma_k}>'
        mma_tile_shape_template = f'cute::Shape<{m_template}, {n_template}, {k_template}>'
        values = {'mma_m': mma_m, 'mma_n': mma_n, 'mma_k': mma_k}
        return Template(mma_tile_shape_template).substitute(values)

    def cluster_shape(self, operation) -> str:
        m_template = 'cute::_${cluster_shape_m}' if operation.tile_description.cluster_shape[0] > 0 else 'int(0)'
        n_template = 'cute::_${cluster_shape_n}' if operation.tile_description.cluster_shape[1] > 0 else 'int(0)'
        k_template = 'cute::_${cluster_shape_k}' if operation.tile_description.cluster_shape[2] > 0 else 'int(0)'
        cluster_shape_template = f'cute::Shape<{m_template}, {n_template}, {k_template}>'
        values = {'cluster_shape_m': operation.tile_description.cluster_shape[0], 'cluster_shape_n': operation.tile_description.cluster_shape[1], 'cluster_shape_k': operation.tile_description.cluster_shape[2]}
        return Template(cluster_shape_template).substitute(values)

    def stage_count(self, operation) -> str:
        namespace_prefix = 'cutlass::conv::collective::'
        if operation.tile_description.stages > 0:
            return f'{namespace_prefix}StageCount<{str(operation.tile_description.stages)}>'
        else:
            return f'{namespace_prefix}StageCountAutoCarveout<sizeof(typename {operation.procedural_name()}_epilogue::SharedStorage)>'

    def emit(self, operation) -> str:
        _LOGGER.debug('*** EmitConv3xInstance::emit')
        _LOGGER.debug('***   operation: procedural_name()=' + operation.procedural_name())
        if not hasattr(operation, 'is_3x') or not operation.is_3x:
            raise RuntimeError('operation must be a CUTLASS 3 operation')
        epi_tile_mn = 'cutlass::epilogue::collective::EpilogueTileAuto'
        opcode_class_main = OpcodeClassTag[operation.tile_description.math_instruction.opcode_class]
        opcode_class_epi = opcode_class_main
        tile_shape = operation.tile_description.tile_shape
        cluster_m = operation.tile_description.cluster_shape[0]
        cluster_n = operation.tile_description.cluster_shape[1]
        cta_m, cta_n, cta_k = tile_shape
        warp_count = operation.tile_description.warp_count
        epilogue_schedule = EpilogueScheduleTag[operation.epilogue_schedule]
        kernel_schedule = KernelScheduleTag[operation.kernel_schedule].replace('gemm::', 'conv::')
        tile_scheduler = TileSchedulerTag[operation.tile_scheduler]
        opcode_class = OpcodeClassTag[operation.tile_description.math_instruction.opcode_class]
        values = {'operation_name': operation.procedural_name(), 'conv_kind': ConvKindTag[operation.conv_kind], 'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(), 'element_a': DataTypeTag[operation.A.element], 'layout_a': LayoutTag[operation.A.layout], 'align_a': int(operation.A.alignment), 'element_b': DataTypeTag[operation.B.element], 'layout_b': LayoutTag[operation.B.layout], 'align_b': int(operation.B.alignment), 'element_c': DataTypeTag[operation.C.element], 'layout_c': LayoutTag[operation.C.layout], 'align_c': int(operation.C.alignment), 'element_d': DataTypeTag[operation.D.element], 'layout_d': LayoutTag[operation.D.layout], 'align_d': int(operation.D.alignment), 'element_accumulator': DataTypeTag[operation.accumulator_type()], 'opcode_class': opcode_class, 'arch': self.arch_number_to_type(operation.arch), 'output_cta_tile_shape': self.output_cta_tile_shape(operation, cta_m, cta_n, cta_k), 'mma_tile_shape': self.mma_tile_shape(operation, cta_m, cta_n, cta_k), 'cluster_shape': self.cluster_shape(operation), 'opcode_class_epi': opcode_class_epi, 'opcode_class_main': opcode_class_main, 'epi_tile_mn': epi_tile_mn, 'stages': self.stage_count(operation), 'kernel_schedule': kernel_schedule, 'epilogue_schedule': epilogue_schedule, 'tile_scheduler': tile_scheduler, 'element_compute': DataTypeTag[operation.element_compute]}
        return Template(self.template).substitute(values)

class EmitConv3xIncludes:

    def __init__(self):
        _LOGGER.debug('*** EmitConv3xIncludes::__init__')
        self.includes = ['conv_operation_3x.hpp', 'cutlass/conv/device/conv_universal_adapter.hpp', 'cutlass/conv/kernel/conv_universal.hpp', 'cutlass/conv/collective/collective_builder.hpp', 'cutlass/epilogue/collective/collective_builder.hpp']

    def emit(self, operation) -> str:
        _LOGGER.debug('*** EmitConv3xIncludes::emit')
        return '\n'.join((f'#include "{incl}"' for incl in self.includes)) + '\n\n///////////////////////////////////////////////////////////////////////////////////////////////////'