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
import re
import sys
from pathlib import Path
from .literal_map import dtype_literal, idtype_literal, mask_mode_literal

def get_cu_file_str(head_dim_qk, head_dim_vo, pos_encoding_mode, use_fp16_qk_reduction, mask_mode, dtype_q, dtype_kv, dtype_out, idtype):
    pos_encoding_mode = None
    use_fp16_qk_reduction = None

    def get_insts(attention_variant):
        return '\ntemplate cudaError_t BatchPrefillWithPagedKVCacheDispatched\n    <{head_dim_qk},\n     {head_dim_vo},\n     {mask_mode},\n     /*USE_SLIDING_WINDOW=*/true,\n     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/true,\n     {attention_variant},\n     Params>\n    (Params& params, cudaStream_t stream);\n\ntemplate cudaError_t BatchPrefillWithPagedKVCacheDispatched\n    <{head_dim_qk},\n     {head_dim_vo},\n     {mask_mode},\n     /*USE_SLIDING_WINDOW=*/true,\n     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/false,\n     {attention_variant},\n     Params>\n    (Params& params, cudaStream_t stream);\n\ntemplate cudaError_t BatchPrefillWithPagedKVCacheDispatched\n    <{head_dim_qk},\n     {head_dim_vo},\n     {mask_mode},\n     /*USE_SLIDING_WINDOW=*/false,\n     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/true,\n     {attention_variant},\n     Params>\n    (Params& params, cudaStream_t stream);\n\ntemplate cudaError_t BatchPrefillWithPagedKVCacheDispatched\n    <{head_dim_qk},\n     {head_dim_vo},\n     {mask_mode},\n     /*USE_SLIDING_WINDOW=*/false,\n     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/false,\n     {attention_variant},\n     Params>\n    (Params& params, cudaStream_t stream);\n    '.format(head_dim_qk=head_dim_qk, head_dim_vo=head_dim_vo, mask_mode=mask_mode_literal[int(mask_mode)], attention_variant=attention_variant)
    dtype_q = dtype_literal[dtype_q]
    dtype_kv = dtype_literal[dtype_kv]
    dtype_out = dtype_literal[dtype_out]
    idtype = idtype_literal[idtype]
    content = f" // batch_paged_prefill_sm90 template inst\n#include <flashinfer/attention/hopper/default_params.cuh>\n#include <flashinfer/attention/hopper/prefill_sm90.cuh>\n#include <flashinfer/attention/hopper/variants.cuh>\n#include <flashinfer/cutlass_utils.cuh>\n\n\nnamespace flashinfer {{\n\nusing DTypeQ = cutlass_dtype_t<{dtype_q}>;\nusing DTypeKV = cutlass_dtype_t<{dtype_kv}>;\nusing DTypeO = cutlass_dtype_t<{dtype_out}>;\n\nusing Params = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, {idtype}>;\n\n{get_insts('LogitsSoftCap')}\n\n{get_insts('StandardAttention')}\n\n}}"
    return content
if __name__ == '__main__':
    pattern = 'batch_paged_prefill_head_qk_([0-9]+)_head_vo_([0-9]+)_posenc_([0-9]+)_fp16qkred_([a-z]+)_mask_([0-9]+)_dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)_sm90\\.cu'
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)
    path.write_text(get_cu_file_str(*match.groups()))