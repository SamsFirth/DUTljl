import kp
import numpy as np

def main():
    mgr = kp.Manager()
    tensor_size = 4
    tensor_shape = [tensor_size, tensor_size]
    tensor_in_1 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_in_2 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_out = mgr.tensor(np.zeros(tensor_shape))
    print(f'Input tensors:\n{tensor_in_1.data().reshape(tensor_shape)}\n{tensor_in_2.data().reshape(tensor_shape)}\n')
    params = [tensor_in_1, tensor_in_2, tensor_out]
    matmul_shader = kp.Shader.compile_source('\n#version 450\n\nlayout (local_size_x = 1, local_size_y = 1) in;\n\nlayout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };\nlayout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };\nlayout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };\n\nlayout (constant_id = 0) const float tensor_size_f = 0;\n\n\nvoid main()\n{\n    uint globalRow = gl_GlobalInvocationID.x;\n    uint globalCol = gl_GlobalInvocationID.y;\n    uint tensor_size = uint(tensor_size_f);\n    float acc = 0.0;\n    for(uint k = 0u; k < tensor_size; k++)\n        acc += in_tensor_1[(k * tensor_size) + globalRow] * in_tensor_2[(globalCol * tensor_size) + k];\n    out_tensor[(globalCol * tensor_size) + globalRow] = acc;\n}')
    algo = mgr.algorithm(params, matmul_shader, (*tensor_shape, 1), [float(tensor_size)], [])
    mgr.sequence().record(kp.OpTensorSyncDevice(params)).record(kp.OpAlgoDispatch(algo)).record(kp.OpTensorSyncLocal(params)).eval()
    print(f'Output :\n{tensor_out.data().reshape(tensor_shape)}')
if __name__ == '__main__':
    main()