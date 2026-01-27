def type_2_cutlass_type(input_type='fp16'):
    if input_type == 'fp32':
        return 'float'
    if input_type == 'bf16':
        return 'cutlass::bfloat16_t'
    if input_type == 'fp16':
        return 'cutlass::half_t'
    if input_type == 'int32':
        return 'int32_t'
    if input_type == 'int8':
        return 'int8_t'
    if input_type == 'Row':
        return 'cutlass::layout::RowMajor'
    if input_type == 'Col':
        return 'cutlass::layout::ColumnMajor'

def cvt_2_cutlass_shape(gemm_shape):
    if len(gemm_shape) == 3:
        val = 'cutlass::gemm::GemmShape<' + str(gemm_shape[0]) + ', ' + str(gemm_shape[1]) + ', ' + str(gemm_shape[2]) + '>'
        return val

def write_2_headfile(filename, file_dir, string):
    with open(file_dir + filename, 'w') as f:
        f.write('/* Auto Generated code - Do not edit.*/\n\n\n#pragma once\n' + string)

def var_idx(varaiable, index):
    return varaiable + str(index)

def list_2_string(input_list):
    rtn_string = ''
    cnt = 0
    for element in input_list:
        final = ', \n'
        if cnt == len(input_list) - 1:
            final = '\n'
        cnt += 1
        rtn_string += str(element) + final
    return rtn_string

def get_epilogue_info(layer_info):
    return layer_info['epilogue']

def get_epilogue_tp(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['tp']

def get_epilogue_add_bias_or_not(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['bias']['addbias']

def get_epilogue_add_bias_tp(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['bias']['bias_tp']

def get_epilogue_args(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['args']

def get_epilogue_bias_shape(layer_info):
    bias_tp = get_epilogue_add_bias_tp(layer_info).lower()
    mn_shape = layer_info['mnk'][:-1]
    if bias_tp == 'mat':
        mn_shape[0] = 'M'
        return mn_shape
    elif bias_tp == 'vec':
        mn_shape[0] = 1
        return mn_shape
    else:
        assert 0

def get_epilogue_bias_ldm(layer_info):
    bias_tp = get_epilogue_add_bias_tp(layer_info).lower()
    mn_shape = layer_info['mnk'][:-1]
    c_layout = layer_info['C_format'].lower()
    if c_layout != 'row':
        assert 0
    if bias_tp == 'mat':
        return mn_shape[1]
    elif bias_tp == 'vec':
        return 0
    else:
        assert 0

def get_epilogue_compute_tp(layer_info):
    return layer_info['Acc_tp']