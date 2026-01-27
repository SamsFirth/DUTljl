"""
Basic example of using the CUTLASS Python interface to run a GEMM
"""
import sys
print('This example is deprecated. Please see examples/python for examples of using the CUTLASS Python interface.')
sys.exit(0)
import argparse
import numpy as np
import cutlass_bindings
import cutlass.backend as pycutlass
from cutlass.backend import *
from cutlass.backend.utils.device import device_cc
parser = argparse.ArgumentParser(description="Launch a GEMM kernel from Python: 'D = alpha * A * B + beta * C'")
parser.add_argument('--m', default=128, type=int, help='M dimension of the GEMM')
parser.add_argument('--n', default=128, type=int, help='N dimension of the GEMM')
parser.add_argument('--k', default=128, type=int, help='K dimension of the GEMM')
parser.add_argument('--print_cuda', action='store_true', help='Print the underlying CUDA kernel')
try:
    args = parser.parse_args()
except:
    sys.exit(0)
cc = device_cc()
assert cc >= 70, 'The CUTLASS Python GEMM example requires compute capability greater than or equal to 70.'
alignment = 8
assert args.m % alignment == 0, 'M dimension of size {} is not divisible by alignment of {}'.format(args.m, alignment)
assert args.n % alignment == 0, 'N dimension of size {} is not divisible by alignment of {}'.format(args.n, alignment)
assert args.k % alignment == 0, 'K dimension of size {} is not divisible by alignment of {}'.format(args.k, alignment)
np.random.seed(0)
pycutlass.get_memory_pool(init_pool_size=2 ** 30, max_pool_size=2 ** 32)
pycutlass.compiler.nvcc()
A = TensorDescription(cutlass_bindings.float16, cutlass_bindings.ColumnMajor, alignment)
B = TensorDescription(cutlass_bindings.float16, cutlass_bindings.RowMajor, alignment)
C = TensorDescription(cutlass_bindings.float32, cutlass_bindings.ColumnMajor, alignment)
element_acc = cutlass_bindings.float32
element_epilogue = cutlass_bindings.float32
if cc == 70:
    instruction_shape = [8, 8, 4]
elif cc == 75:
    instruction_shape = [16, 8, 8]
else:
    cc = 80
    instruction_shape = [16, 8, 16]
math_inst = MathInstruction(instruction_shape, A.element, B.element, element_acc, cutlass_bindings.OpClass.TensorOp, MathOperation.multiply_add)
tile_description = TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst)
epilogue_functor = pycutlass.LinearCombination(C.element, C.alignment, element_acc, element_epilogue)
operation = GemmOperationUniversal(arch=cc, tile_description=tile_description, A=A, B=B, C=C, epilogue_functor=epilogue_functor)
if args.print_cuda:
    print(operation.rt_module.emit())
operations = [operation]
pycutlass.compiler.add_module(operations)
tensor_A = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(args.m * args.k,))).astype(np.float16)
tensor_B = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(args.k * args.n,))).astype(np.float16)
tensor_C = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(args.m * args.n,))).astype(np.float32)
tensor_D = np.zeros(shape=(args.m * args.n,)).astype(np.float32)
problem_size = cutlass_bindings.gemm.GemmCoord(args.m, args.n, args.k)
alpha = 1.0
beta = 0.0
arguments = GemmArguments(operation=operation, problem_size=problem_size, A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D, output_op=operation.epilogue_type(alpha, beta))
operation.run(arguments)
arguments.sync()
reference = ReferenceModule(A, B, C)
tensor_D_ref = reference.run(tensor_A, tensor_B, tensor_C, problem_size, alpha, beta)
try:
    assert np.array_equal(tensor_D, tensor_D_ref)
except:
    assert np.allclose(tensor_D, tensor_D_ref, atol=1e-05)
print('Passed.')