"""
Utilities for generating source for building a PyTorch CUDA extension that using a CUTLASS kernel.
If specified, the extension can be JIT compiled via PyTorch's ``cpp_extension.load`` method.

Example usage with JIT compilation:

.. highlight:: python
.. code-block:: python

    plan = cutlass.op.Gemm(element=torch.float32, layout=cutlass_library.LayoutType.RowMajor)
    op = plan.construct()
    mod = cutlass.emit.pytorch(op, 'cutlass_gemm', 80, jit=True)

    # Generate inputs for the GEMM
    A, B, C = [torch.ones((512, 512)).to('cuda') for _ in range(3)]

    # Run the module
    D = mod.run(A, B, C)


Example usage without JIT compilation:

.. highlight:: python
.. code-block:: python

    plan = cutlass.op.Gemm(element=torch.float32, layout=cutlass.LayoutType.RowMajor)
    op = plan.construct()
    cutlass.emit.pytorch(op, 'cutlass_gemm', 80, jit=False, sourcedir='output')

After this call, the directory ``output`` contains ``setup.py``,
``cutlass_gemm.cpp``, and ``cutlass_gemm_kernel.cu``. The module can be built from
within ``output`` by running: ``TORCH_CUDA_ARCH_LIST="8.0" python setup.py develop --user``.

The module can later be used in Python via:

.. highlight:: python
.. code-block:: python

    import torch
    import cutlass_gemm

    # Generate inputs for the GEMM
    A, B, C = [torch.ones((512, 512)).to('cuda') for _ in range(3)]

    # Run the module
    D = cutlass_gemm.run(A, B, C)
"""
import logging
import os
from cutlass_library import ConvKind, ConvKindNames, DataType, SubstituteTemplate
from cutlass import CUTLASS_PATH, logger, swizzle
from cutlass.backend.gemm_operation import GemmOperationGrouped, GemmOperationUniversal
from cutlass.backend.conv2d_operation import Conv2dOperation
from cutlass.backend.library import ApiVersion
from cutlass.emit import common
from cutlass.utils.datatypes import is_torch_available
if is_torch_available():
    import torch
_PYTORCH_CUDA_TEMPLATE = common._CSTYLE_AUTOGEN_COMMENT + '\n#include <cuda_runtime.h>\n#include <torch/extension.h>\n#include <ATen/ATen.h>\n#include <ATen/cuda/CUDAContext.h>\n#include "cutlass/cutlass.h"\n#include "cutlass/util/device_memory.h"\n\n// helper function allocating the memory\nvoid* device_memory_allocation(size_t size, int device_id=0) {\n    if (size > 0) {\n        torch::Device device(torch::kCUDA, device_id);\n        cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kI8).device(device);\n        at::Tensor device_tensor = torch::empty({(long)size,}, options);\n        return reinterpret_cast<void*>(device_tensor.data_ptr());\n    } else {\n        return nullptr;\n    }\n}\n\n${includes}\n${declaration}\n${impl}\n'
_PYTORCH_GEMM_CPP_TEMPLATE = common._CSTYLE_AUTOGEN_COMMENT + '\n#include <torch/extension.h>\n#include <ATen/ATen.h>\n#include <pybind11/stl.h>\n\n// CUDA forward declarations\nat::Tensor ${name}_kernel(const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt, float alpha=1.f, float beta=0.f);\n\n// C++ interface\nat::Tensor ${name}(const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt, float alpha=1.f, float beta=0.f) {\n  return ${name}_kernel(A, B, C, alpha, beta);\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("run", py::overload_cast<const at::Tensor&, const at::Tensor&, at::optional<const at::Tensor>, float, float>(&${name}), py::arg("A"), py::arg("B"), py::arg("C") = nullptr, py::arg("alpha") = 1.f, py::arg("beta") = 0.f);\n}\n'
_PYTORCH_GROUPED_GEMM_CPP_TEMPLATE = common._CSTYLE_AUTOGEN_COMMENT + '\n#include <torch/extension.h>\n#include <ATen/ATen.h>\n#include <pybind11/stl.h>\n\n// CUDA forward declarations\nstd::vector<at::Tensor> ${name}_kernel(const std::vector<at::Tensor>& A, const std::vector<at::Tensor>& B, at::optional<const std::vector<at::Tensor>> C=at::nullopt, float alpha=1.f, float beta=0.f);\n\n// C++ interface\nstd::vector<at::Tensor> ${name}(const std::vector<at::Tensor>& A, const std::vector<at::Tensor>& B, at::optional<const std::vector<at::Tensor>> C=at::nullopt, float alpha=1.f, float beta=0.f) {\n  return ${name}_kernel(A, B, C, alpha, beta);\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("run", py::overload_cast<const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, at::optional<const std::vector<at::Tensor>>, float, float>(&${name}),\n        py::arg("A"), py::arg("B"), py::arg("C") = nullptr, py::arg("alpha") = 1.f, py::arg("beta") = 0.f);\n}\n'
_PYTORCH_CONV2D_FPROP_CPP_TEMPLATE = common._CSTYLE_AUTOGEN_COMMENT + '\n#include <torch/extension.h>\n#include <ATen/ATen.h>\n#include <pybind11/stl.h>\n\n// CUDA forward declarations\nat::Tensor ${name}_kernel(\n    const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},\n    float alpha=1.f, float beta=0.f,\n    std::string split_k_mode="serial", int split_k_slices=1);\n\n// C++ interface\nat::Tensor ${name}(\n    const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},\n    float alpha=1.f, float beta=0.f,\n    std::string split_k_mode="serial", int split_k_slices=1) {\n    return ${name}_kernel(A, B, C, stride, padding, dilation, alpha, beta, split_k_mode, split_k_slices);\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("run",\n  py::overload_cast<\n    const at::Tensor&, const at::Tensor&, at::optional<const at::Tensor>,\n    std::tuple<int, int>, std::tuple<int, int>, std::tuple<int, int>, float, float,  std::string, int>(\n        &${name}), py::arg("A"), py::arg("B"), py::arg("C") = nullptr,\n        py::arg("stride") = std::make_tuple(1, 1), py::arg("padding") = std::make_tuple(1, 1), py::arg("dilation") = std::make_tuple(1, 1),\n        py::arg("alpha") = 1.f, py::arg("beta") = 0.f,\n        py::arg("split_k_mode") = "serial", py::arg("split_k_slices") = 1);\n}\n'
_PYTORCH_CONV2D_GRAD_CPP_TEMPLATE = common._CSTYLE_AUTOGEN_COMMENT + '\n#include <torch/extension.h>\n#include <ATen/ATen.h>\n#include <pybind11/stl.h>\n\n// CUDA forward declarations\nat::Tensor ${name}_kernel(\n    std::tuple<int, int, int, int> result_size, const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},\n    float alpha=1.f, float beta=0.f,\n    std::string split_k_mode="serial", int split_k_slices=1);\n\n// C++ interface\nat::Tensor ${name}(\n    std::tuple<int, int, int, int> result_size, const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},\n    float alpha=1.f, float beta=0.f,\n    std::string split_k_mode="serial", int split_k_slices=1) {\n    return ${name}_kernel(result_size, A, B, C, stride, padding, dilation, alpha, beta, split_k_mode, split_k_slices);\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("run",\n  py::overload_cast<\n    std::tuple<int, int, int, int>, const at::Tensor&, const at::Tensor&, at::optional<const at::Tensor>,\n    std::tuple<int, int>, std::tuple<int, int>, std::tuple<int, int>, float, float, std::string, int>(\n        &${name}), py::arg("result_size"), py::arg("A"), py::arg("B"), py::arg("C") = nullptr,\n        py::arg("stride") = std::make_tuple(1, 1), py::arg("padding") = std::make_tuple(1, 1), py::arg("dilation") = std::make_tuple(1, 1),\n        py::arg("alpha") = 1.f, py::arg("beta") = 0.f,\n        py::arg("split_k_mode") = "serial", py::arg("split_k_slices") = 1);\n}\n'
_PYTORCH_GEMM_INCLUDES = {ApiVersion.v2x: '\n#include "cutlass/gemm/device/gemm_universal.h"\n', ApiVersion.v3x: '\n#include "cutlass/gemm/device/gemm_universal_adapter.h"\n#include "cutlass/gemm/collective/collective_builder.hpp"\n#include "cutlass/gemm/device/gemm_universal_adapter.h"\n#include "cutlass/gemm/kernel/gemm_universal.hpp"\n#include "cutlass/epilogue/collective/collective_builder.hpp"\n#include "cutlass/util/packed_stride.hpp"\n'}
_PYTORCH_GROUPED_GEMM_INCLUDES = '\n#include "cutlass/gemm/kernel/default_gemm_grouped.h"\n#include "cutlass/gemm/device/gemm_grouped.h"\n'
_PYTORCH_CONV2D_INCLUDES = '\n#include "cutlass/conv/kernel/default_conv2d_fprop.h"\n#include "cutlass/conv/kernel/default_conv2d_dgrad.h"\n#include "cutlass/conv/kernel/default_conv2d_wgrad.h"\n#include "cutlass/conv/device/implicit_gemm_convolution.h"\n'
_CUTLASS_TYPE_TO_TORCH_TYPE = {DataType.f16: 'torch::kF16', DataType.f32: 'torch::kF32', DataType.f64: 'torch::kF64', DataType.s8: 'torch::kI8', DataType.s32: 'torch::kI32'}
_PYTORCH_GEMM_IMPL_TEMPLATE_2x = common._CUTLASS_KERNEL_RUN_GEMM_2x + '\nat::Tensor ${name}_kernel(const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C, float alpha, float beta) {\n    int M = A.size(0);\n    int N = B.size(1);\n    int K = A.size(1);\n\n    typename DeviceKernel::ElementC* ptrC = (C == at::nullopt) ?\n                                            nullptr :\n                                            reinterpret_cast<typename DeviceKernel::ElementC*>(C->contiguous().data_ptr());\n    at::Tensor D = B.new_empty({M, N}, ${torch_type_C});\n\n    cutlass::Status status = ${name}_kernel_run(M, N, K,\n                                                reinterpret_cast<typename DeviceKernel::ElementA*>(A.contiguous().data_ptr()),\n                                                reinterpret_cast<typename DeviceKernel::ElementB*>(B.contiguous().data_ptr()),\n                                                ptrC,\n                                                reinterpret_cast<typename DeviceKernel::ElementC*>(D.contiguous().data_ptr()),\n                                                ElementCompute(alpha), ElementCompute(beta));\n\n    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");\n    return D;\n}\n'
_PYTORCH_GEMM_IMPL_TEMPLATE_3x = common._CUTLASS_KERNEL_RUN_GEMM_3x + '\nbool hw_info_queried = false;\ncutlass::KernelHardwareInfo hw_info;\n\nat::Tensor ${name}_kernel(const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C, float alpha, float beta) {\n    int M = A.size(0);\n    int N = B.size(1);\n    int K = A.size(1);\n    int L = 1;\n\n    // Query hardware info if we haven\'t already\n    if (!hw_info_queried) {\n        hw_info.device_id = 0;\n        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);\n    }\n\n    typename DeviceKernel::ElementC* ptrC = (C == at::nullopt) ?\n                                            nullptr :\n                                            reinterpret_cast<typename DeviceKernel::ElementC*>(C->contiguous().data_ptr());\n    at::Tensor D = B.new_empty({M, N}, ${torch_type_C});\n\n    cutlass::Status status = ${name}_kernel_run(M, N, K, L,\n                                                reinterpret_cast<typename DeviceKernel::ElementA*>(A.contiguous().data_ptr()),\n                                                reinterpret_cast<typename DeviceKernel::ElementB*>(B.contiguous().data_ptr()),\n                                                ptrC,\n                                                reinterpret_cast<typename DeviceKernel::ElementC*>(D.contiguous().data_ptr()),\n                                                ElementCompute(alpha), ElementCompute(beta),\n                                                hw_info);\n\n    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");\n    return D;\n}\n'
_PYTORCH_GROUPED_GEMM_IMPL_TEMPLATE = common._CUTLASS_KERNEL_RUN_GROUPED_GEMM_2x + '\nstd::vector<at::Tensor> ${name}_kernel(const std::vector<at::Tensor>& A, const std::vector<at::Tensor>& B, at::optional<const std::vector<at::Tensor>> C, float alpha, float beta) {\n    size_t num = A.size();\n\n    // To avoid performing many small cudaMallocs and host-to-device copies,\n    // we serialize the grouped GEMM arguments on the host, allocate one\n    // large chunk of device memory, and perform a single cudaMemcpy to\n    // copy the host data to the device. Allocation overheads could be\n    // avoided by using a memory pool.\n\n    // Calculate the total size of the data to be copied from host to device\n    size_t total_size = sizeof(cutlass::gemm::GemmCoord) +\n                        sizeof(DeviceKernel::ElementA*) +\n                        sizeof(DeviceKernel::ElementB*) +\n                        sizeof(DeviceKernel::ElementC*) +\n                        sizeof(DeviceKernel::ElementC*) +\n                        sizeof(int64_t) +\n                        sizeof(int64_t) +\n                        sizeof(int64_t);\n    total_size *= num;\n\n    // num * sizeof(cutlass::gemm::GemmCoord) may leave one at a non-multiple\n    // of sizeof(DeviceKernel::ElementA*) (which will be 64 on a 64-bit system).\n    // To ensure that we don\'t end up having misaligned loads in the kernel,\n    // we pad to the nearest multiple of 8.\n    //\n    // Note that, even on a 32-bit system (for which sizeof(X*) will not equal\n    // sizeof(int64_t)), only padding between the list of GemmCoords and the\n    // list of ptr_As is sufficient because the set of four equal-length lists of pointers\n    // (A*, B*, C*, D*) will ensure that the first list of int64_ts will always\n    // start on a multiple of 8.\n    int64_t padding = 8 - (total_size % 8);\n    total_size += padding;\n\n    uint8_t* host_data = new uint8_t[total_size];\n    cutlass::DeviceAllocation<uint8_t> device_data(total_size);\n\n    uint8_t* start = host_data;\n    cutlass::gemm::GemmCoord* problem_sizes_host = reinterpret_cast<cutlass::gemm::GemmCoord*>(start);\n\n    // Apply the padding after the list of GemmCoords\n    start += num * sizeof(cutlass::gemm::GemmCoord) + padding;\n\n    int64_t ptr_A_offset = start - host_data;\n    DeviceKernel::ElementA** ptr_A_host = reinterpret_cast<DeviceKernel::ElementA**>(start);\n    start += num * sizeof(DeviceKernel::ElementA*);\n\n    int64_t ptr_B_offset = start - host_data;\n    DeviceKernel::ElementB** ptr_B_host = reinterpret_cast<DeviceKernel::ElementB**>(start);\n    start += num * sizeof(DeviceKernel::ElementB*);\n\n    int64_t ptr_C_offset = start - host_data;\n    DeviceKernel::ElementC** ptr_C_host = reinterpret_cast<DeviceKernel::ElementC**>(start);\n    start += num * sizeof(DeviceKernel::ElementC*);\n\n    int64_t ptr_D_offset = start - host_data;\n    DeviceKernel::ElementC** ptr_D_host = reinterpret_cast<DeviceKernel::ElementC**>(start);\n    start += num * sizeof(DeviceKernel::ElementC*);\n\n    int64_t lda_offset = start - host_data;\n    int64_t* lda_host = reinterpret_cast<int64_t*>(start);\n    start += num * sizeof(int64_t);\n\n    int64_t ldb_offset = start - host_data;\n    int64_t* ldb_host = reinterpret_cast<int64_t*>(start);\n    start += num * sizeof(int64_t);\n\n    int64_t ldc_offset = start - host_data;\n    int64_t* ldc_host = reinterpret_cast<int64_t*>(start);\n    start += num * sizeof(int64_t);\n\n    std::vector<at::Tensor> D(num);\n\n    bool need_C = (C != at::nullopt) && (beta != 0.f);\n    for (size_t i = 0; i < num; ++i) {\n        int M = A[i].size(0);\n        int N = B[i].size(1);\n        int K = A[i].size(1);\n        *(problem_sizes_host + i) = {M, N, K};\n        *(ptr_A_host + i) = reinterpret_cast<typename DeviceKernel::ElementA*>(A[i].contiguous().data_ptr());\n        *(ptr_B_host + i) = reinterpret_cast<typename DeviceKernel::ElementB*>(B[i].contiguous().data_ptr());\n\n        if (need_C) {\n            *(ptr_C_host + i) = reinterpret_cast<typename DeviceKernel::ElementC*>(C->at(i).contiguous().data_ptr());\n        }\n        else {\n            *(ptr_C_host + i) = nullptr;\n        }\n\n        D[i] = B[i].new_empty({M, N}, ${torch_type_C});\n        *(ptr_D_host + i) = reinterpret_cast<typename DeviceKernel::ElementC*>(D[i].contiguous().data_ptr());\n\n        *(lda_host + i) = DeviceKernel::LayoutA::packed({M, K}).stride(0);\n        *(ldb_host + i) = DeviceKernel::LayoutB::packed({K, N}).stride(0);\n        *(ldc_host + i) = DeviceKernel::LayoutC::packed({M, N}).stride(0);\n    }\n\n    device_data.copy_from_host(host_data);\n\n    cutlass::Status status = ${name}_kernel_run(\n        num,\n        reinterpret_cast<cutlass::gemm::GemmCoord*>(device_data.get()),\n        reinterpret_cast<DeviceKernel::ElementA**>(device_data.get() + ptr_A_offset),\n        reinterpret_cast<DeviceKernel::ElementB**>(device_data.get() + ptr_B_offset),\n        reinterpret_cast<DeviceKernel::ElementC**>(device_data.get() + ptr_C_offset),\n        reinterpret_cast<DeviceKernel::ElementC**>(device_data.get() + ptr_D_offset),\n        reinterpret_cast<int64_t*>(device_data.get() + lda_offset),\n        reinterpret_cast<int64_t*>(device_data.get() + ldb_offset),\n        reinterpret_cast<int64_t*>(device_data.get() + ldc_offset),\n        reinterpret_cast<int64_t*>(device_data.get() + ldc_offset),\n        ElementCompute(alpha), ElementCompute(beta));\n\n    delete[] host_data;\n\n    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");\n    return D;\n}\n'
_PYTORCH_CONV2D_IMPL_TEMPLATE_2x = '\n    cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n\n    cutlass::Status status = ${name}_kernel_run(\n        &problem_size,\n        reinterpret_cast<typename UnderlyingKernel::ElementA*>(A.data_ptr()),\n        reinterpret_cast<typename UnderlyingKernel::ElementB*>(B.data_ptr()),\n        ptrC,\n        reinterpret_cast<typename UnderlyingKernel::ElementC*>(D.data_ptr()),\n        alpha, beta,\n        split_k_mode, stream, B.device().index());\n\n    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");\n    return D;\n}\n'
_PYTORCH_CONV2D_FPROP_IMPL_TEMPLATE_2x = common._CUTLASS_KERNEL_RUN_CONV2D_2x + '\nat::Tensor ${name}_kernel(const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},\n    float alpha=1.f, float beta=0.f, std::string split_k_mode="serial", int split_k_slices=1) {\n    int N, H, W, C_, K, R, S, P, Q;\n    N = A.size(0);\n    C_ = A.size(1);\n    H = A.size(2);\n    W = A.size(3);\n\n    K = B.size(0);\n    R = B.size(2);\n    S = B.size(3);\n\n    cutlass::conv::Conv2dProblemSize problem_size(\n        cutlass::Tensor4DCoord(N, H, W, C_),\n        cutlass::Tensor4DCoord(K, R, S, C_),\n        cutlass::Tensor4DCoord(std::get<0>(padding), std::get<0>(padding), std::get<1>(padding), std::get<1>(padding)),\n        cutlass::MatrixCoord(std::get<0>(stride), std::get<1>(stride)),\n        cutlass::MatrixCoord(std::get<0>(dilation), std::get<1>(dilation)),\n        cutlass::conv::Mode::kCrossCorrelation,\n        split_k_slices\n    );\n\n    P = problem_size.P;\n    Q = problem_size.Q;\n\n    typename UnderlyingKernel::ElementC* ptrC = (C == at::nullopt) ?\n                                            nullptr :\n                                            reinterpret_cast<typename UnderlyingKernel::ElementC*>(C->data_ptr());\n\n    torch::TensorOptions options = torch::TensorOptions().dtype(${torch_type_C}).device(B.device()).memory_format(at::MemoryFormat::ChannelsLast);\n    at::Tensor D = torch::zeros({N, K, P, Q}, options);\n' + _PYTORCH_CONV2D_IMPL_TEMPLATE_2x
_PYTORCH_CONV2D_DGRAD_IMPL_TEMPLATE_2x = common._CUTLASS_KERNEL_RUN_CONV2D_2x + '\nat::Tensor ${name}_kernel(std::tuple<int, int, int, int> input_size, const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1}, float alpha=1.f, float beta=0.f,\n    std::string split_k_mode="serial", int split_k_slices=1) {\n    int N, H, W, C_, K, R, S;\n    N = std::get<0>(input_size);\n    C_ = std::get<1>(input_size);\n    H = std::get<2>(input_size);\n    W = std::get<3>(input_size);\n\n    K = B.size(0);\n    R = B.size(2);\n    S = B.size(3);\n\n    cutlass::conv::Conv2dProblemSize problem_size(\n        cutlass::Tensor4DCoord(N, H, W, C_),\n        cutlass::Tensor4DCoord(K, R, S, C_),\n        cutlass::Tensor4DCoord(std::get<0>(padding), std::get<0>(padding), std::get<1>(padding), std::get<1>(padding)),\n        cutlass::MatrixCoord(std::get<0>(stride), std::get<1>(stride)),\n        cutlass::MatrixCoord(std::get<0>(dilation), std::get<1>(dilation)),\n        cutlass::conv::Mode::kCrossCorrelation,\n        split_k_slices\n    );\n\n    typename UnderlyingKernel::ElementC* ptrC = (C == at::nullopt) ?\n                                            nullptr :\n                                            reinterpret_cast<typename UnderlyingKernel::ElementC*>(C->data_ptr());\n\n    torch::TensorOptions options = torch::TensorOptions().dtype(${torch_type_C}).device(B.device()).memory_format(at::MemoryFormat::ChannelsLast);\n    at::Tensor D = torch::empty({N, C_, H, W}, options);\n' + _PYTORCH_CONV2D_IMPL_TEMPLATE_2x
_PYTORCH_CONV2D_WGRAD_IMPL_TEMPLATE_2x = common._CUTLASS_KERNEL_RUN_CONV2D_2x + '\nat::Tensor ${name}_kernel(std::tuple<int, int, int, int> weight_size, const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,\n    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1}, float alpha=1.f, float beta=0.f,\n    std::string split_k_mode="serial", int split_k_slices=1) {\n    int N, H, W, C_, K, R, S;\n    K = std::get<0>(weight_size);\n    C_ = std::get<1>(weight_size);\n    R = std::get<2>(weight_size);\n    S = std::get<3>(weight_size);\n\n    N = B.size(0);\n    H = B.size(2);\n    W = B.size(3);\n\n    cutlass::conv::Conv2dProblemSize problem_size(\n        cutlass::Tensor4DCoord(N, H, W, C_),\n        cutlass::Tensor4DCoord(K, R, S, C_),\n        cutlass::Tensor4DCoord(std::get<0>(padding), std::get<0>(padding), std::get<1>(padding), std::get<1>(padding)),\n        cutlass::MatrixCoord(std::get<0>(stride), std::get<1>(stride)),\n        cutlass::MatrixCoord(std::get<0>(dilation), std::get<1>(dilation)),\n        cutlass::conv::Mode::kCrossCorrelation,\n        split_k_slices\n    );\n\n    typename UnderlyingKernel::ElementC* ptrC = (C == at::nullopt) ?\n                                            nullptr :\n                                            reinterpret_cast<typename UnderlyingKernel::ElementC*>(C->data_ptr());\n\n    torch::TensorOptions options = torch::TensorOptions().dtype(${torch_type_C}).device(B.device()).memory_format(at::MemoryFormat::ChannelsLast);\n    at::Tensor D = torch::empty({K, C_, R, S}, options);\n' + _PYTORCH_CONV2D_IMPL_TEMPLATE_2x
_PYTORCH_SETUP_PY = common._PYSTYLE_AUTOGEN_COMMENT + "\nfrom setuptools import setup\nfrom torch.utils.cpp_extension import BuildExtension, CUDAExtension\n\nsetup(\n    name='${name}',\n    ext_modules=[\n        CUDAExtension('${name}', [\n            '${name}.cpp',\n            '${name}_kernel.cu',\n        ],\n        include_dirs=['${cutlass_path}/include', '${cutlass_path}/tools/util/include'],\n        extra_compile_args={\n            'cxx': ['-std=c++17'],\n            'nvcc': ['-std=c++17', ${extra_compile_args}],\n        },\n        libraries=['cuda']\n        ),\n    ],\n    cmdclass={\n        'build_ext': BuildExtension\n    })\n\n"

def _generate_setup(name: str, sourcedir: str, extra_compile_args: str=''):
    """
    Generates a setup.py file for the extension

    :param name: name of the module to generate
    :type name: str
    :param sourcedir: directory to which generated source files should be written
    :type sourcedir: str
    :param extra_compile_args: additional arguments to pass to setup.py
    :type extra_args: str
    """
    setup_py_file = os.path.join(sourcedir, 'setup.py')
    setup_source = SubstituteTemplate(_PYTORCH_SETUP_PY, {'name': name, 'cutlass_path': CUTLASS_PATH, 'extra_compile_args': extra_compile_args})
    with open(setup_py_file, 'w') as outfile:
        outfile.write(setup_source)

class _ArchListSetter:
    """
    Utility context manager for temporarily setting the value of the ``TORCH_CUDA_ARCH_LIST``
    environment variable when building a PyTorch CUDA module.

    ``TORCH_CUDA_ARCH_LIST`` is a space-delmited list of compute capabilites for which a PyTorch
    CUDA module should be compiled.

    For example, ``TORCH_CUDA_ARCH_LIST="7.0 8.0"`` would result in the inclusion of
    ``-gencode=arch=compute_70,code=sm_70`` and ``-gencode=arch=compute_80,code=sm_80`` in the
    compilation of the module.

    This utility wraps the building of a PyTorch CUDA module with a setting of this environment
    variable according to the current compute capability being targetted.

    Example usage:

    .. highlight:: python
    .. code-block:: python

        # Temporarily set TORCH_CUDA_ARCH_LIST="8.0"
        with _ArchListSetter(80):
            # Perform JIT compilation and loading of the module
            mod = torch.utils.cpp_extension.load(...)

    :param cc: compute capability
    :type cc: int
    """
    _TORCH_CUDA_ARCH_LIST = 'TORCH_CUDA_ARCH_LIST'

    def __init__(self, cc: int):
        self.cc_str = '.'.join(list(str(cc)))

    def __enter__(self):
        """
        Saves the old value of TORCH_CUDA_ARCH_LIST and reset it to the new value based on ``cc``
        """
        self.old_arch_list = os.getenv(_ArchListSetter._TORCH_CUDA_ARCH_LIST)
        os.environ[_ArchListSetter._TORCH_CUDA_ARCH_LIST] = self.cc_str
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """
        Restores the old value of TORCH_CUDA_ARCH_LIST
        """
        if self.old_arch_list is None:
            del os.environ[_ArchListSetter._TORCH_CUDA_ARCH_LIST]
        else:
            os.environ[_ArchListSetter._TORCH_CUDA_ARCH_LIST] = self.old_arch_list

def _jit(name: str, cc: int, cpp_file: str, cuda_file: str):
    """
    JIT compiles and loads a PyTorch CUDA extension.

    :param name: name of the module to generate
    :type name: str
    :param cc: compute capability of the device the module should target
    :type cc: int
    :param cpp_file: path to file containing extension's C++ interface
    :type cpp_file: str
    :param cuda_file: path to file containing extension's CUDA interface
    :type cuda_file: str

    :return: loaded PyTorch module
    """
    from torch.utils.cpp_extension import load
    extra_cuda_cflags = ['-std=c++17']
    if cc == 90:
        extra_cuda_cflags.append('-gencode=arch=compute_90a,code=sm_90a')
    with _ArchListSetter(cc):
        jitmodule = load(name, [cpp_file, cuda_file], extra_cuda_cflags=extra_cuda_cflags, extra_include_paths=[os.path.join(CUTLASS_PATH, 'include'), os.path.join(CUTLASS_PATH, 'tools/util/include')], extra_ldflags=['-lcuda'], verbose=logger.level == logging.DEBUG)
    return jitmodule

def _pytorch_gemm(op, name: str, cc: int, jit: bool=False, sourcedir: str=''):
    """
    Generates source for building a PyTorch CUDA module that leverages the CUTLASS GEMM
    specified by ``op``. If the ``jit`` parameter is set to true, the module is just-in-time
    compiled, loaded, and returned.

    :param op: operation to emit in the module
    :param name: name of the module to generate
    :type name: str
    :param cc: compute capability of the device the module should target
    :type cc: int
    :param jit: whether the module should be just-in-time compiled
    :type jit: bool
    :param sourcedir: directory to which generated source files should be written
    :type sourcedir: str

    :return: loaded PyTorch module if ``jit=True`` or ``None`` otherwise
    """
    if sourcedir != '' and (not os.path.isdir(sourcedir)):
        os.makedirs(sourcedir)
    cuda_file = os.path.join(sourcedir, name + '_kernel.cu')
    extra_kw = {}
    if op.api == ApiVersion.v3x:
        impl_template = _PYTORCH_GEMM_IMPL_TEMPLATE_3x
    else:
        impl_template = _PYTORCH_GEMM_IMPL_TEMPLATE_2x
        if op.swizzling_functor == swizzle.ThreadblockSwizzleStreamK:
            extra_kw['args'] = common._CUTLASS_KERNEL_ARGS_2x_STREAM_K
        else:
            extra_kw['args'] = common._CUTLASS_KERNEL_ARGS_2x
    impl_template = _PYTORCH_GEMM_IMPL_TEMPLATE_3x if op.api == ApiVersion.v3x else _PYTORCH_GEMM_IMPL_TEMPLATE_2x
    cuda_impl = SubstituteTemplate(impl_template, {'name': name, **extra_kw})
    cuda_source = SubstituteTemplate(_PYTORCH_CUDA_TEMPLATE, {'includes': _PYTORCH_GEMM_INCLUDES[op.api], 'declaration': op.rt_module.emit(), 'procedural_name': op.procedural_name(), 'impl': cuda_impl, 'torch_type_C': _CUTLASS_TYPE_TO_TORCH_TYPE[op.C.element]})
    with open(cuda_file, 'w') as outfile:
        outfile.write(cuda_source)
    cpp_file = os.path.join(sourcedir, name + '.cpp')
    cpp_source = SubstituteTemplate(_PYTORCH_GEMM_CPP_TEMPLATE, {'name': name, 'description': f'CUTLASS {op.procedural_name()} GEMM'})
    with open(cpp_file, 'w') as outfile:
        outfile.write(cpp_source)
    extra_compile_args = ''
    if cc == 90:
        extra_compile_args = "'--generate-code=arch=compute_90a,code=[sm_90a]'"
    _generate_setup(name, sourcedir, extra_compile_args)
    if jit:
        return _jit(name, cc, cpp_file, cuda_file)
    return None

def _pytorch_grouped_gemm(op, name: str, cc: int, jit: bool=False, sourcedir: str=''):
    """
    Generates source for building a PyTorch CUDA module that leverages the CUTLASS grouped GEMM
    specified by ``op``. If the ``jit`` parameter is set to true, the module is just-in-time
    compiled, loaded, and returned.

    :param op: operation to emit in the module
    :param name: name of the module to generate
    :type name: str
    :param cc: compute capability of the device the module should target
    :type cc: int
    :param jit: whether the module should be just-in-time compiled
    :type jit: bool
    :param sourcedir: directory to which generated source files should be written
    :type sourcedir: str

    :return: loaded PyTorch module if ``jit=True`` or ``None`` otherwise
    """
    if op.api != ApiVersion.v2x:
        raise Exception('Grouped GEMM is currently only supported for CUTLASS 2.x')
    if sourcedir != '' and (not os.path.isdir(sourcedir)):
        os.makedirs(sourcedir)
    cuda_file = os.path.join(sourcedir, name + '_kernel.cu')
    cuda_impl = SubstituteTemplate(_PYTORCH_GROUPED_GEMM_IMPL_TEMPLATE, {'name': name})
    cuda_source = SubstituteTemplate(_PYTORCH_CUDA_TEMPLATE, {'includes': _PYTORCH_GROUPED_GEMM_INCLUDES, 'declaration': op.rt_module.emit(), 'procedural_name': op.procedural_name(), 'impl': cuda_impl, 'torch_type_C': _CUTLASS_TYPE_TO_TORCH_TYPE[op.C.element]})
    with open(cuda_file, 'w') as outfile:
        outfile.write(cuda_source)
    cpp_file = os.path.join(sourcedir, name + '.cpp')
    cpp_source = SubstituteTemplate(_PYTORCH_GROUPED_GEMM_CPP_TEMPLATE, {'name': name, 'description': f'CUTLASS {op.procedural_name()} grouped GEMM'})
    with open(cpp_file, 'w') as outfile:
        outfile.write(cpp_source)
    _generate_setup(name, sourcedir)
    if jit:
        return _jit(name, cc, cpp_file, cuda_file)
    return None

def _pytorch_conv2d(op, name: str, cc: int, jit: bool=False, sourcedir: str=''):
    """
    Generates source for building a PyTorch CUDA module that leverages the CUTLASS Conv2d
    specified by ``op``. If the ``jit`` parameter is set to true, the module is just-in-time
    compiled, loaded, and returned.

    :param op: operation to emit in the module
    :param name: name of the module to generate
    :type name: str
    :param cc: compute capability of the device the module should target
    :type cc: int
    :param jit: whether the module should be just-in-time compiled
    :type jit: bool
    :param sourcedir: directory to which generated source files should be written
    :type sourcedir: str

    Note that the when conv kind is `dgrad` or `wgrad`, the size of the input `(N, C, H, W)` or
    weight `(K, C, R, S)` should be provided. This is because there are multiple valid solutions
    for H/W/R/S given the same P/Q.

    :return: loaded PyTorch module if ``jit=True`` or ``None`` otherwise
    """
    if sourcedir != '' and (not os.path.isdir(sourcedir)):
        os.makedirs(sourcedir)
    cuda_file = os.path.join(sourcedir, name + '_kernel.cu')
    extra_kw = {}
    if op.conv_kind == ConvKind.Fprop:
        impl_template = _PYTORCH_CONV2D_FPROP_IMPL_TEMPLATE_2x
        cpp_template = _PYTORCH_CONV2D_FPROP_CPP_TEMPLATE
    elif op.conv_kind == ConvKind.Dgrad:
        impl_template = _PYTORCH_CONV2D_DGRAD_IMPL_TEMPLATE_2x
        cpp_template = _PYTORCH_CONV2D_GRAD_CPP_TEMPLATE
    elif op.conv_kind == ConvKind.Wgrad:
        impl_template = _PYTORCH_CONV2D_WGRAD_IMPL_TEMPLATE_2x
        cpp_template = _PYTORCH_CONV2D_GRAD_CPP_TEMPLATE
    extra_kw['conv_kind_name'] = ConvKindNames[op.conv_kind].capitalize()
    extra_kw['torch_type_C'] = _CUTLASS_TYPE_TO_TORCH_TYPE[op.C.element]
    cuda_impl = SubstituteTemplate(impl_template, {'name': name, **extra_kw})
    cuda_source = SubstituteTemplate(_PYTORCH_CUDA_TEMPLATE, {'includes': _PYTORCH_CONV2D_INCLUDES, 'declaration': op.rt_module.emit(), 'procedural_name': op.procedural_name(), 'impl': cuda_impl, 'torch_type_C': _CUTLASS_TYPE_TO_TORCH_TYPE[op.C.element]})
    with open(cuda_file, 'w') as outfile:
        outfile.write(cuda_source)
    cpp_file = os.path.join(sourcedir, name + '.cpp')
    cpp_source = SubstituteTemplate(cpp_template, {'name': name, 'description': f'CUTLASS {op.procedural_name()} Conv2d'})
    with open(cpp_file, 'w') as outfile:
        outfile.write(cpp_source)
    _generate_setup(name, sourcedir)
    if jit:
        return _jit(name, cc, cpp_file, cuda_file)
    return None

def pytorch(op, name: str, cc: int, jit: bool=False, sourcedir: str=''):
    """
    Generates source for building a PyTorch CUDA module that leverages the CUTLASS kernel
    specified by ``op``. If the ``jit`` parameter is set to true, the module is just-in-time
    compiled, loaded, and returned.

    The result of this method is files within ``sourcedir`` that can be used for building
    a PyTorch module.

    :param op: operation to emit in the module
    :param name: name of the module to generate
    :type name: str
    :param cc: compute capability of the device the module should target
    :type cc: int
    :param jit: whether the module should be just-in-time compiled
    :type jit: bool
    :param sourcedir: directory to which generated source files should be written
    :type sourcedir: str

    :return: loaded PyTorch module (if ``jit=True``) or None
    """
    device_op = op.device_op()
    if isinstance(op, GemmOperationUniversal):
        return _pytorch_gemm(device_op, name, cc, jit, sourcedir)
    elif isinstance(op, GemmOperationGrouped):
        return _pytorch_grouped_gemm(device_op, name, cc, jit, sourcedir)
    elif isinstance(op, Conv2dOperation):
        return _pytorch_conv2d(device_op, name, cc, jit, sourcedir)
    else:
        raise Exception(f'Operation type {type(op)} is not currently supported for PyTorch emission.')