#include "gptq_marlin/ops.h"
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
// namespace py = pybind11;

PYBIND11_MODULE(vLLMMarlin, m) {

    m.def("gptq_marlin_gemm", &gptq_marlin_gemm,
          "Function to perform GEMM using Marlin quantization.", py::arg("a"),
          py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
          py::arg("perm"), py::arg("workspace"), py::arg("num_bits"), py::arg("size_m_tensor"),
          py::arg("size_m"), py::arg("size_n"), py::arg("size_k"),
          py::arg("sms"), py::arg("is_k_full"));
    m.def("gptq_marlin_repack", &gptq_marlin_repack,
            "gptq_marlin repack from GPTQ");
}