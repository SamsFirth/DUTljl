class gen_build_sys:

    def __init__(self, cutlass_deps_dir, output_dir='../'):
        self.output_dir = output_dir
        self.cutlass_deps_dir = cutlass_deps_dir

    def gen_top(self):
        code = ''
        code += '# Auto Generated code - Do not edit.\n\ncmake_minimum_required(VERSION 3.8)\nproject(CUTLASS_MULTI_GEMMS LANGUAGES CXX CUDA)\nfind_package(CUDAToolkit)\nset(CUDA_PATH ${{CUDA_TOOLKIT_ROOT_DIR}})\nset(CUTLASS_PATH "{cutlass_deps_dir}/include")\nset(CUTLASS_UTIL_PATH "{cutlass_deps_dir}/tools/util/include")\nlist(APPEND CMAKE_MODULE_PATH ${{CUDAToolkit_LIBRARY_DIR}})\n'.format(cutlass_deps_dir=self.cutlass_deps_dir)
        code += 'set(GPU_ARCHS "" CACHE STRING\n  "List of GPU architectures (semicolon-separated) to be compiled for.")\n\nif("${GPU_ARCHS}" STREQUAL "")\n\tset(GPU_ARCHS "70")\nendif()\n\nforeach(arch ${GPU_ARCHS})\n  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")\n\tif(SM STREQUAL 70 OR SM STREQUAL 75)\n    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")\n    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")\n    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")\n\tendif()\nendforeach()\n\nset(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")\nset(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")\nset(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")\n\nset(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")\nset(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")\nset(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")\n\nset(CMAKE_CXX_STANDARD 11)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n\nif(CMAKE_CXX_STANDARD STREQUAL "11")\n  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")\n  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")\nendif()\n\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")\nset(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")\nset(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")\n\nset(COMMON_HEADER_DIRS\n  ${PROJECT_SOURCE_DIR}\n  ${CUDAToolkit_INCLUDE_DIRS}\n)\n\nset(COMMON_LIB_DIRS\n  ${CUDAToolkit_LIBRARY_DIR}\n)\nlist(APPEND COMMON_HEADER_DIRS ${CUTLASS_PATH})\nlist(APPEND COMMON_HEADER_DIRS ${CUTLASS_UTIL_PATH})\n'
        code += 'include_directories(\n  ${COMMON_HEADER_DIRS}\n)\n\nlink_directories(\n  ${COMMON_LIB_DIRS}\n)\n\nadd_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)\nadd_definitions(-DGOOGLE_CUDA=1)\n\nadd_executable(sample\n  sample/sample.cu\n  one_api.cu\n)\ntarget_link_libraries(sample PRIVATE\n  -lcudart\n  -lnvToolsExt\n  ${CMAKE_THREAD_LIBS_INIT}\n)\n\nif(NOT DEFINED LIB_INSTALL_PATH)\n\tset(LIB_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR})\nendif()\n'
        return code

    def gen_code(self):
        top_code = self.gen_top()
        with open(self.output_dir + 'CMakeLists.txt', 'w') as f:
            f.write(top_code)