"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import jinja2
from .core import load_cuda_ops
from .env import FLASHINFER_GEN_SRC_DIR
from .utils import write_if_different
activation_templ = '\n#include <flashinfer/activation.cuh>\n#include "pytorch_extension_utils.h"\n#include <cuda_runtime.h>\n\n{% set func_name = act_func_name ~ \'_and_mul\' %}\n\nusing namespace flashinfer;\n\n{{ act_func_def }}\n\nvoid {{ func_name }}(at::Tensor& out, at::Tensor& input, bool enable_pdl) {\n  int d = input.size(-1) / 2;\n  int64_t num_tokens = input.numel() / input.size(-1);\n  dim3 grid(num_tokens);\n\n  const c10::cuda::OptionalCUDAGuard device_guard(out.device());\n  auto stream = at::cuda::getCurrentCUDAStream();\n  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {\n    uint32_t vec_size = 16 / sizeof(c_type);\n    cudaLaunchConfig_t config;\n    config.gridDim = num_tokens;\n    config.blockDim = std::min(d / vec_size, 1024U);\n    config.dynamicSmemBytes = 0;\n    config.stream = stream;\n    cudaLaunchAttribute attrs[1];\n    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;\n    config.numAttrs = 1;\n    config.attrs = attrs;\n\n    auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}>;\n\n    cudaLaunchKernelEx(&config, kernel, static_cast<c_type*>(out.data_ptr()),\n                       static_cast<c_type*>(input.data_ptr()), d);\n\n    cudaError_t err = cudaGetLastError();\n    TORCH_CHECK(err == cudaSuccess, "Failed to launch kernel: ", cudaGetErrorString(err));\n\n    return true;\n  });\n}\n\nTORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {\n  m.def("{{ func_name }}", {{ func_name }});\n}\n'

def get_act_and_mul_cu_str(act_func_name: str, act_func_def: str) -> str:
    template = jinja2.Template(activation_templ)
    return template.render(act_func_name=act_func_name, act_func_def=act_func_def)

def gen_act_and_mul_module(act_func_name: str, act_func_def: str) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    os.makedirs(gen_directory, exist_ok=True)
    sources = [gen_directory / f'{act_func_name}_and_mul.cu']
    write_if_different(sources[0], get_act_and_mul_cu_str(act_func_name, act_func_def))
    return load_cuda_ops(f'{act_func_name}_and_mul', sources)