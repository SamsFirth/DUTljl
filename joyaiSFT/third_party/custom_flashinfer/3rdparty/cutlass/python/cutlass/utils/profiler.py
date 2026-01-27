"""
Profiler based on the cuda events
"""
import re
import subprocess
from cuda import cuda, cudart
import numpy as np
from cutlass import CUTLASS_PATH
from cutlass.backend.library import DataTypeSize
from cutlass.op.op import OperationBase
from cutlass.shape import GemmCoord
from cutlass.utils.datatypes import is_numpy_tensor

class GpuTimer:

    def __init__(self) -> None:
        self.events = [cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1], cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]]

    def start(self, stream=cuda.CUstream(0)):
        err, = cuda.cuEventRecord(self.events[0], stream)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'CUDA Error {str(err)}')

    def stop(self, stream=cuda.CUstream(0)):
        err, = cuda.cuEventRecord(self.events[1], stream)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'CUDA Error {str(err)}')
        pass

    def stop_and_wait(self, stream=cuda.CUstream(0)):
        self.stop(stream)
        if stream:
            err, = cuda.cuStreamSynchronize(stream)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'CUDA Error {str(err)}')
        else:
            err, = cudart.cudaDeviceSynchronize()
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'CUDA Error {str(err)}')

    def duration(self, iterations=1):
        err, duration = cuda.cuEventElapsedTime(self.events[0], self.events[1])
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'CUDA Error {str(err)}')
        return duration / float(iterations)

class CUDAEventProfiler:

    def __init__(self, op: OperationBase, warmup_iterations: int=500, iterations: int=500, *args, **kwargs) -> None:
        self.arguments = op.run(*args, **kwargs)
        self.operation = op.operation
        self.warmup_iterations = warmup_iterations
        self.iterations = iterations
        self.timer = GpuTimer()

    def __call__(self):
        for _ in range(self.warmup_iterations):
            self.operation.run(self.arguments)
        self.timer.start()
        for _ in range(self.iterations):
            self.operation.run(self.arguments)
        self.timer.stop_and_wait()
        runtime = self.timer.duration(self.iterations)
        return runtime

    def run_cutlass_profiler(self):
        alpha = 1.0
        beta = 1.0
        profiler_path = CUTLASS_PATH + '/build/tools/profiler/cutlass_profiler'
        kernel_name = self.operation.procedural_name()
        verification_providers = 'device'
        provider = 'cutlass'
        problem_size = self.arguments.problem_size
        if 'cutlass3x' in kernel_name:
            layout_name = self.operation.layout_name_3x()
            if layout_name[-1] == 't':
                new_layout_name = ''.join(['n' for l in layout_name if l == 't' or 't'])
                problem_size = GemmCoord(problem_size.n, problem_size.m, problem_size.k)
                kernel_name = kernel_name.replace(layout_name, new_layout_name)
        batch_count = self.arguments.batch_count
        cmd = f'{profiler_path} --kernels={kernel_name} --verification-providers={verification_providers} --providers={provider} --m={problem_size.m()} --n={problem_size.n()} --k={problem_size.k()} --batch_count={batch_count} --alpha={alpha} --beta={beta} --warmup-iterations={self.warmup_iterations} --profiling-iterations={self.iterations}'
        result = subprocess.getoutput(cmd)
        m = re.search('Runtime:\\s+(?P<runtime>\\d+.\\d+)', result)
        runtime = float(m.group('runtime'))
        m = re.search('Bytes:\\s+(?P<bytes>\\d+)', result)
        bytes = int(m.group('bytes'))
        m = re.search('FLOPs:\\s+(?P<flops>\\d+)', result)
        flops = int(m.group('flops'))
        assert bytes == self.bytes(problem_size, batch_count, beta)
        assert flops == self.flops(problem_size, batch_count, beta)
        return runtime

    def bytes(self, problem_size, batch_count=1, beta=0.0):
        m = problem_size.m()
        n = problem_size.n()
        k = problem_size.k()
        bytes = DataTypeSize[self.operation.A.element] * m // 8 * k + DataTypeSize[self.operation.B.element] * n // 8 * k + DataTypeSize[self.operation.C.element] * m // 8 * n
        if beta != 0:
            bytes += DataTypeSize[self.operation.C.element] * m // 8 * n
        bytes *= batch_count
        return bytes

    def flops(self, problem_size, batch_count=1, beta=0.0):
        m = problem_size.m()
        n = problem_size.n()
        k = problem_size.k()
        flops_ = m * n * k * 2 * batch_count
        if beta != 0:
            flops_ += m * n * batch_count * 2
        return flops_