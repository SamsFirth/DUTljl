import time
import kp
import numpy as np

class MatMulOp:

    def __init__(self, manager: kp.Manager, tile_size: int=-1, thread_work_ratio: int=16):
        self.mgr = manager
        props = self.mgr.get_device_properties()
        max_workgroup_invocation = props['max_work_group_invocations']
        max_workgroup_size = props['max_work_group_size']
        if tile_size < 0:
            tile_size = 1
            local_size_y = tile_size // thread_work_ratio
            while 4 * tile_size * tile_size <= max_workgroup_invocation and 2 * tile_size <= max_workgroup_size[0] and (2 * tile_size <= max_workgroup_size[1]):
                tile_size *= 2
                local_size_y = tile_size // thread_work_ratio
        else:
            local_size_y = tile_size // thread_work_ratio
        assert tile_size > 0
        assert thread_work_ratio > 0
        assert tile_size * local_size_y <= max_workgroup_invocation
        assert tile_size <= max_workgroup_size[0]
        assert local_size_y <= max_workgroup_size[1]
        self.tile_size = tile_size
        self.thread_work_ratio = thread_work_ratio
        self.local_size_x = tile_size
        self.local_size_y = tile_size // thread_work_ratio
        self.shader = '\n#version 450\n\nlayout (local_size_x = {tile_size}, local_size_y = {local_size_y}) in;\n\nlayout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};\nlayout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};\nlayout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};\n\nlayout (constant_id = 0) const float tensor_size_f = 0;\n\nshared float sub_tensor_1[{tile_size}][{tile_size}];\nshared float sub_tensor_2[{tile_size}][{tile_size}];\n\nvoid main()\n{{\n    uint row = gl_LocalInvocationID.x;\n    uint col = gl_LocalInvocationID.y;\n    uint globalRow = {tile_size} * gl_WorkGroupID.x + row;\n    uint globalCol = {tile_size} * gl_WorkGroupID.y + col;\n\n    uint tensor_size = uint(tensor_size_f);\n    float acc[{thread_work_ratio}];\n    for(uint w = 0u; w < {thread_work_ratio}; w++)\n        acc[w] = 0.0;\n\n    uint numTiles = tensor_size / {tile_size};\n    for(uint t = 0u; t < numTiles; t++)\n    {{\n        for(uint w = 0u; w < {thread_work_ratio}; w++)\n        {{\n            uint tiledRow = {tile_size} * t + row;\n            uint tiledCol = {tile_size} * t + col;\n            sub_tensor_1[col + w * {local_size_y}][row] = in_tensor_1[\n                (tiledCol + w * {local_size_y}) * tensor_size + globalRow];\n            sub_tensor_2[col + w * {local_size_y}][row] = in_tensor_2[\n                (globalCol + w * {local_size_y})* tensor_size + tiledRow];\n        }}\n\n        memoryBarrierShared();\n        barrier();\n\n        for(uint k = 0u; k < {tile_size}; k++)\n            for(uint w = 0u; w < {thread_work_ratio}; w++)\n                acc[w] += sub_tensor_1[k][row] * sub_tensor_2[col + w * {local_size_y}][k];\n\n        barrier();\n    }}\n    for(uint w = 0u; w < {thread_work_ratio}; w++)\n        out_tensor[(globalCol + w * {local_size_y}) * tensor_size + globalRow] = acc[w];\n}}'
        self.compiled_shader = kp.Shader.compile_source(self.shader.format(tile_size=tile_size, thread_work_ratio=thread_work_ratio, local_size_y=local_size_y))
        self.tensor_shape: tuple[int, int] = (0, 0)
        self.params: list[kp.Tensor] = []
        self.algo = None

    def __call__(self, tensor_shape: tuple[int, int], tensor_in_1: kp.Tensor, tensor_in_2: kp.Tensor, tensor_out: kp.Tensor):
        params = [tensor_in_1, tensor_in_2, tensor_out]
        if self.algo is None or self.tensor_shape != tensor_shape or self.params != params:
            self.tensor_shape = tensor_shape
            self.params = params
            tile_size = min(self.tensor_shape[0], self.tile_size)
            thread_work_ratio = min(self.tensor_shape[1] // self.tile_size, self.thread_work_ratio)
            local_size_y = tile_size // thread_work_ratio
            self.compiled_shader = kp.Shader.compile_source(self.shader.format(tile_size=tile_size, thread_work_ratio=thread_work_ratio, local_size_y=local_size_y))
            workgroup = (tensor_shape[0] // self.local_size_x, tensor_shape[1] // self.local_size_y, 1)
            self.algo = self.mgr.algorithm(params, self.compiled_shader, workgroup, [float(tensor_shape[0])], [])
        self.mgr.sequence().record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2])).record(kp.OpAlgoDispatch(self.algo)).record(kp.OpTensorSyncLocal([tensor_out])).eval()

def main():
    mgr = kp.Manager()
    matmul_op = MatMulOp(mgr)
    tensor_size = 4096
    tensor_shape = [tensor_size, tensor_size]
    tensor_in_1 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_in_2 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_out = mgr.tensor(np.zeros(tensor_shape))
    print(f'{tensor_shape} input tensors:\n{tensor_in_1.data().reshape(tensor_shape)}\n{tensor_in_2.data().reshape(tensor_shape)}\n')
    matmul_op(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)
    experiment_count = 2
    start_time = time.time()
    for _ in range(experiment_count):
        matmul_op(tensor_shape, tensor_in_1, tensor_in_2, tensor_out)
    end_time = time.time()
    experiment_time = end_time - start_time
    op_count = tensor_shape[0] * tensor_shape[1] * (tensor_shape[1] * 2 - 1)
    print(f'Output :\n{tensor_out.data().reshape(tensor_shape)}')
    print(f'{experiment_count} matmul time : {experiment_time * 1000:0.2f}ms => {experiment_count / experiment_time:0.2f}op/s or {experiment_count * op_count / (1000000000.0 * experiment_time):0.2f} GFLOPS')
if __name__ == '__main__':
    main()