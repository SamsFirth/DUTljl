from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from joyaiSFT.operators.base_operator import BaseInjectedModule
from tqdm import tqdm
import time
import logging
from tqdm.auto import tqdm
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'joyaiSFT_ext', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'joyaiSFT_ext', 'build', 'Release'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'joyaiSFT_ext', 'build', 'Debug'))
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
from cpuinfer_ext.sft_moe import SFT_MOEConfig, SFT_MOE
import ctypes
from joyaiSFT.util.custom_loader import GGUFLoader
from joyaiSFT.util.inference_state import InferenceState
from joyaiSFT.util.custom_gguf import GGMLQuantizationType
from joyaiSFT.util.custom_loader import GGUFLoader, SafeTensorLoader, ModelLoader
from joyaiSFT.server.config.config import Config
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from joyaiSFT.operators.linear import JOYLinearMarlin, JOYLinearTorch, JOYSFTLinear
import time
from joyaiSFT.operators.cpuinfer import CPUInfer
from joyaiSFT.util.grad_wrapper import maybe_no_grad
H_FIXED = 7168
M_FIXED = 2048

def deduplicate_and_sort(lst):
    return sorted(set(lst))

def generate_cuda_graphs(chunk_size: int) -> list:
    assert chunk_size <= 1024 or chunk_size % 1024 == 0, 'chunk_size must <= 1024 or a multiple of 1024'
    base_list = [1, 2, 3, Config().max_batch_size, 64, 256, 512, chunk_size]
    if chunk_size <= 1024:
        return deduplicate_and_sort(base_list)
    multiples = [i for i in range(1024, chunk_size + 1, 1024)]
    return deduplicate_and_sort(base_list + multiples)
if torch.cuda.is_available():
    cuda_graphs = generate_cuda_graphs(Config().chunk_size)
else:
    cuda_graphs = 1

class JOYExpertsBase(ABC):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str='cuda', **kwargs):
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device

    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str='cpu', warmup: bool=False):
        pass

    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None=None, device: str='cpu'):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None
        for key in keys:
            if self.gguf_loader.has_tensor(key + '.ffn_gate_exps.weight'):
                targets = ['.ffn_gate_exps.weight', '.ffn_up_exps.weight', '.ffn_down_exps.weight']
                tensors = self.load_multi(key, targets, device=device)
                gate = tensors['.ffn_gate_exps.weight']
                up = tensors['.ffn_up_exps.weight']
                down = tensors['.ffn_down_exps.weight']
                gate_type = self.gguf_loader.tensor_info[key + '.ffn_gate_exps.weight']['ggml_type']
                up_type = self.gguf_loader.tensor_info[key + '.ffn_up_exps.weight']['ggml_type']
                down_type = self.gguf_loader.tensor_info[key + '.ffn_down_exps.weight']['ggml_type']
            elif self.gguf_loader.has_tensor(key + '.ffn_down.0.weight'):
                gate = []
                up = []
                down = []
                for i in range(8):
                    gatei, upi, downi = (f'.ffn_gate.{i}.weight', f'.ffn_up.{i}.weight', f'.ffn_down.{i}.weight')
                    targets = [gatei, upi, downi]
                    tensors = self.load_multi(key, targets, device=device)
                    gate_it, up_it, down_it = (tensors[gatei], tensors[upi], tensors[downi])
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = torch.stack(gate)
                up = torch.stack(up)
                down = torch.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + '.ffn_gate.0.weight']['ggml_type']
                up_type = self.gguf_loader.tensor_info[key + '.ffn_up.0.weight']['ggml_type']
                down_type = self.gguf_loader.tensor_info[key + '.ffn_down.0.weight']['ggml_type']
            else:
                raise ValueError(f'Experts {key} not found in gguf_loader')
            res = {key: {'gate': gate, 'up': up, 'down': down, 'gate_type': gate_type, 'up_type': up_type, 'down_type': down_type}}
        return res

    def load_multi(self, key: str, keys: list[str], device: str='cpu'):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors

class JOYExpertsCPU(JOYExpertsBase):
    input_tensor_cpu: Tensor = None
    expert_ids_cpu: Tensor = None
    weights_cpu: Tensor = None
    output_cpu: Tensor = None
    output_gpu_map: dict = {}
    CPU_INFER = CPUInfer(Config().cpu_infer)

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, n_routed_experts: int, orig_module: nn.Module=None, device: str='cpu', out_device: str='cuda', **kwargs):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        assert device.lower() == 'cpu', 'JOYExpertsCPU can only be loaded on CPU'
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device
        self.backend = kwargs.get('backend', 'llamafile')

    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str | None=None, warmup: bool=False):
        if device:
            assert device.lower() == 'cpu', 'JOYExpertsCPU can only be loaded on CPU, Parameter "device" can be cpu or None.'
        if w is None:
            w = self.load_weights()[self.key]
        self.gate = w['gate']
        self.up = w['up']
        self.down = w['down']
        self.gate_type = w['gate_type']
        self.up_type = w['up_type']
        self.down_type = w['down_type']
        gate_ptr = ctypes.addressof(ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
        up_ptr = ctypes.addressof(ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
        down_ptr = ctypes.addressof(ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
        n_routed_experts = self.n_routed_experts
        self.cpu_infer = JOYExpertsCPU.CPU_INFER
        model_dtype = torch.get_default_dtype()
        if torch.xpu.is_available() and model_dtype == torch.float16:
            hidden_type = 1
        else:
            hidden_type = 30
        if self.backend == 'llamafile':
            moe_config = MOEConfig(n_routed_experts, self.config.num_experts_per_tok, self.config.hidden_size, self.config.moe_intermediate_size, 64, 10, 1024, gate_ptr, up_ptr, down_ptr, self.gate_type, self.up_type, self.down_type, hidden_type)
            self.moe = MOE(moe_config)
        elif self.backend == 'AMXBF16':
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXBF16_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(n_routed_experts, self.config.num_experts_per_tok, self.config.hidden_size, self.config.moe_intermediate_size, max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size, gate_ptr, up_ptr, down_ptr)
            self.moe = AMXBF16_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        elif self.backend == 'AMXInt8':
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXInt8_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(n_routed_experts, self.config.num_experts_per_tok, self.config.hidden_size, self.config.moe_intermediate_size, max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size, gate_ptr, up_ptr, down_ptr)
            self.moe = AMXInt8_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        num_experts_per_tok = self.config.num_experts_per_tok
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in JOYExpertsCPU.output_gpu_map:
            if isinstance(cuda_graphs, list):
                JOYExpertsCPU.output_gpu_map[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device=self.out_device) for i in range(len(cuda_graphs))]
            else:
                JOYExpertsCPU.output_gpu_map[self.out_device] = torch.zeros((cuda_graphs, self.config.hidden_size), device=self.out_device)
        if JOYExpertsCPU.input_tensor_cpu == None:
            if isinstance(cuda_graphs, list):
                JOYExpertsCPU.input_tensor_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device='cpu', pin_memory=True) for i in range(len(cuda_graphs))]
                JOYExpertsCPU.expert_ids_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device='cpu', dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                JOYExpertsCPU.weights_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device='cpu', dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]
                JOYExpertsCPU.output_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device='cpu', pin_memory=True, dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                JOYExpertsCPU.bsz_tensor_cpu = [torch.zeros(1, device='cpu', dtype=torch.int32, pin_memory=True) for i in range(len(cuda_graphs))]
            else:
                JOYExpertsCPU.input_tensor_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device='cpu', pin_memory=True)
                JOYExpertsCPU.expert_ids_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device='cpu', dtype=torch.long, pin_memory=True)
                JOYExpertsCPU.weights_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device='cpu', dtype=torch.float32, pin_memory=True)
                if torch.xpu.is_available():
                    JOYExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device='cpu', pin_memory=True, dtype=model_dtype)
                    JOYExpertsCPU.bsz_tensor_cpu = torch.ones(1, device='cpu', dtype=torch.int32, pin_memory=True)
                else:
                    JOYExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device='cpu', pin_memory=True, dtype=torch.bfloat16)
                    JOYExpertsCPU.bsz_tensor_cpu = torch.zeros(1, device='cpu', dtype=torch.int32, pin_memory=True)

    def submit_for_one_decode(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if bsz_tensor is None:
            bsz_tensor = torch.ones(1, device=input_tensor.device, dtype=torch.int32)
        if cuda_graph_idx != -1:
            JOYExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
            JOYExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
            JOYExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
            JOYExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, self.moe.forward(1, expert_ids.size(-1), JOYExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
        else:
            JOYExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            JOYExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            JOYExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            JOYExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, self.moe.forward(1, expert_ids.size(-1), JOYExpertsCPU.expert_ids_cpu.data_ptr(), JOYExpertsCPU.weights_cpu.data_ptr(), JOYExpertsCPU.input_tensor_cpu.data_ptr(), JOYExpertsCPU.output_cpu.data_ptr(), JOYExpertsCPU.bsz_tensor_cpu.data_ptr()))

    def sync_for_one_decode(self, cuda_graph_idx=0):
        if cuda_graph_idx != -1:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            JOYExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(JOYExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
            return JOYExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]
        else:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            JOYExpertsCPU.output_gpu_map[self.out_device].copy_(JOYExpertsCPU.output_cpu, non_blocking=True)
            return JOYExpertsCPU.output_gpu_map[self.out_device]

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if bsz_tensor is None and (not torch.xpu.is_available() or input_tensor.size(0) > 1):
            bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32)
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            if cuda_graph_idx != -1:
                JOYExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
                JOYExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
                JOYExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
                JOYExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward(expert_ids.size(0), expert_ids.size(-1), JOYExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), JOYExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                JOYExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(JOYExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
                return JOYExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]
            else:
                JOYExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
                JOYExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
                JOYExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
                JOYExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward(expert_ids.size(0), expert_ids.size(-1), JOYExpertsCPU.expert_ids_cpu.data_ptr(), JOYExpertsCPU.weights_cpu.data_ptr(), JOYExpertsCPU.input_tensor_cpu.data_ptr(), JOYExpertsCPU.output_cpu.data_ptr(), JOYExpertsCPU.bsz_tensor_cpu.data_ptr()))
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                JOYExpertsCPU.output_gpu_map[self.out_device].copy_(JOYExpertsCPU.output_cpu, non_blocking=True)
                return JOYExpertsCPU.output_gpu_map[self.out_device]
        elif input_tensor.size(0) == 1 and torch.xpu.is_available():
            JOYExpertsCPU.input_tensor_cpu.copy_(input_tensor.view(-1), non_blocking=True)
            JOYExpertsCPU.expert_ids_cpu.copy_(expert_ids.view(-1), non_blocking=True)
            JOYExpertsCPU.weights_cpu.copy_(weights.view(-1), non_blocking=True)
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), JOYExpertsCPU.expert_ids_cpu.data_ptr(), JOYExpertsCPU.weights_cpu.data_ptr(), JOYExpertsCPU.input_tensor_cpu.data_ptr(), JOYExpertsCPU.output_cpu.data_ptr(), JOYExpertsCPU.bsz_tensor_cpu.data_ptr()))
            self.cpu_infer.sync()
            JOYExpertsCPU.output_gpu_map[self.out_device].copy_(JOYExpertsCPU.output_cpu, non_blocking=True)
            return JOYExpertsCPU.output_gpu_map[self.out_device].view(1, -1)
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            bsz_tensor = bsz_tensor.contiguous().cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr(), bsz_tensor.data_ptr()))
            self.cpu_infer.sync()
            return output.to(device=object.__getattribute__(self, 'out_device'))

    def unload(self):
        return

    def load_weights(self, override_key: str | None=None, device: str='cpu'):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None
        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif self.gguf_loader.has_tensor(key + '.ffn_gate_exps.weight'):
                gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
                gate_type = self.gguf_loader.get_ggml_type(key + '.ffn_gate_exps.weight')
                up_type = self.gguf_loader.get_ggml_type(key + '.ffn_up_exps.weight')
                down_type = self.gguf_loader.get_ggml_type(key + '.ffn_down_exps.weight')
            elif key + '.ffn_gate_exps.weight' in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
                gate_type = self.gguf_loader.tensor_info[key + '.ffn_gate_exps.weight']['ggml_type']
                up_type = self.gguf_loader.tensor_info[key + '.ffn_up_exps.weight']['ggml_type']
                down_type = self.gguf_loader.tensor_info[key + '.ffn_down_exps.weight']['ggml_type']
            elif key + '.ffn_down.0.weight' in self.gguf_loader.tensor_info:
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f'{key}.ffn_gate.{i}.weight')
                    up_it = self.gguf_loader.get_mmap_tensor(f'{key}.ffn_up.{i}.weight')
                    down_it = self.gguf_loader.get_mmap_tensor(f'{key}.ffn_down.{i}.weight')
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.get_ggml_type(key + '.ffn_gate.0.weight')
                up_type = self.gguf_loader.get_ggml_type(key + '.ffn_up.0.weight')
                down_type = self.gguf_loader.get_ggml_type(key + '.ffn_down.0.weight')
            else:
                raise ValueError(f'Experts {key} not found in gguf_loader')
            res = {key: {'gate': gate, 'up': up, 'down': down, 'gate_type': gate_type, 'up_type': up_type, 'down_type': down_type}}
        return res

class JOYSFTExpertsCPU(torch.autograd.Function):
    input_tensor_cpu: Tensor = None
    expert_ids_cpu: Tensor = None
    weights_cpu: Tensor = None
    output_cpu: Tensor = None
    output_gpu_map: dict = {}
    CPU_INFER = CPUInfer(Config().cpu_infer)

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, n_routed_experts: int, orig_module: nn.Module=None, device: str='cpu', out_device: str='cuda', **kwargs):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.gguf_loader = gguf_loader
        assert device.lower() == 'cpu', 'JOYExpertsCPU can only be loaded on CPU'
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device
        self.backend = kwargs.get('backend', 'llamafile')
        self.key = key
        self.config = config
        self.device = device
        self.call_count = 0
        self.flops_per_call = []
        self.times = []
        self.tflops_fwd = []
        self.tflops_bwd = []

    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str | None=None, warmup: bool=False):
        if device:
            assert device.lower() == 'cpu', 'JOYSFTExpertsCPU can only be loaded on CPU, Parameter "device" can be cpu or None.'
        if w is None:
            w = self.load_weights()[self.key]
        self.gate = w['gate']
        self.up = w['up']
        self.down = w['down']
        self.gate_type = w['gate_type']
        self.up_type = w['up_type']
        self.down_type = w['down_type']
        gate_ptr = ctypes.addressof(ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
        up_ptr = ctypes.addressof(ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
        down_ptr = ctypes.addressof(ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
        n_routed_experts = self.n_routed_experts
        self.cpu_infer = JOYSFTExpertsCPU.CPU_INFER
        model_dtype = torch.get_default_dtype()
        if torch.xpu.is_available() and model_dtype == torch.float16:
            hidden_type = 1
        else:
            hidden_type = 30
        if self.backend == 'llamafile':
            moe_config = SFT_MOEConfig(n_routed_experts, self.config.num_experts_per_tok, self.config.hidden_size, self.config.moe_intermediate_size, 64, 10, 1024, gate_ptr, up_ptr, down_ptr, self.gate_type, self.up_type, self.down_type, hidden_type)
            self.moe = SFT_MOE(moe_config)
        elif self.backend == 'AMXBF16':
            from cpuinfer_ext.sft_moe import SFT_AMX_MOEConfig, SFT_AMXBF16_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = SFT_AMX_MOEConfig(n_routed_experts, self.config.num_experts_per_tok, self.config.hidden_size, self.config.moe_intermediate_size, max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size, gate_ptr, up_ptr, down_ptr)
            self.moe = SFT_AMXBF16_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        elif self.backend == 'AMXInt8':
            from cpuinfer_ext.sft_moe import SFT_AMX_MOEConfig, SFT_AMXInt8_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = SFT_AMX_MOEConfig(n_routed_experts, self.config.num_experts_per_tok, self.config.hidden_size, self.config.moe_intermediate_size, max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size, gate_ptr, up_ptr, down_ptr)
            self.moe = SFT_AMXInt8_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        num_experts_per_tok = self.config.num_experts_per_tok
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in JOYSFTExpertsCPU.output_gpu_map:
            JOYSFTExpertsCPU.output_gpu_map[self.out_device] = torch.zeros(self.config.hidden_size, device=self.out_device)
        if JOYSFTExpertsCPU.input_tensor_cpu == None:
            JOYSFTExpertsCPU.input_tensor_cpu = torch.zeros(self.config.hidden_size, device='cpu', pin_memory=True)
            JOYSFTExpertsCPU.expert_ids_cpu = torch.zeros(num_experts_per_tok, device='cpu', dtype=torch.long, pin_memory=True)
            JOYSFTExpertsCPU.weights_cpu = torch.zeros(num_experts_per_tok, device='cpu', dtype=torch.float32, pin_memory=True)
            JOYSFTExpertsCPU.output_cpu = torch.zeros(self.config.hidden_size, device='cpu', pin_memory=True, dtype=torch.bfloat16)
        self.gate = None
        self.up = None
        self.down = None

    def submit_for_one_decode(self, input_tensor, expert_ids, weights):
        JOYSFTExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
        JOYSFTExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
        JOYSFTExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
        self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, self.moe.forward(1, expert_ids.size(0), JOYSFTExpertsCPU.expert_ids_cpu.data_ptr(), JOYSFTExpertsCPU.weights_cpu.data_ptr(), JOYSFTExpertsCPU.input_tensor_cpu.data_ptr(), JOYSFTExpertsCPU.output_cpu.data_ptr()))

    def sync_for_one_decode(self):
        self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
        JOYSFTExpertsCPU.output_gpu_map[self.out_device].copy_(JOYSFTExpertsCPU.output_cpu, non_blocking=True)
        return JOYSFTExpertsCPU.output_gpu_map[self.out_device]

    @staticmethod
    def forward(ctx, input_tensor, expert_ids, weights, cpu_infer, moe, out_device, layer_idx):
        if input_tensor.size(0) == 1 and torch.cuda.is_current_stream_capturing():
            JOYSFTExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            JOYSFTExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            JOYSFTExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, moe.forward(1, expert_ids.size(1), JOYSFTExpertsCPU.expert_ids_cpu.data_ptr(), JOYSFTExpertsCPU.weights_cpu.data_ptr(), JOYSFTExpertsCPU.input_tensor_cpu.data_ptr(), JOYSFTExpertsCPU.output_cpu.data_ptr()))
            cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
            t_fwd = time.time() - wall_t0
            JOYSFTExpertsCPU.output_gpu_map[out_device].copy_(JOYSFTExpertsCPU.output_cpu, non_blocking=True)
            result = JOYSFTExpertsCPU.output_gpu_map[out_device]
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            output = torch.empty_like(input_tensor).contiguous()
            wall_t0 = time.time()
            cpu_infer.submit(moe.forward(expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr()))
            cpu_infer.sync()
            t_fwd = time.time() - wall_t0
            result = output.to(device=out_device)
        ctx.save_for_backward(input_tensor, expert_ids, weights)
        ctx.cpu_infer = cpu_infer
        ctx.moe = moe
        ctx.out_device = out_device
        ctx.layer_idx = layer_idx
        qlen = expert_ids.size(0)
        k = expert_ids.size(1)
        flops_fwd = 6 * qlen * k * H_FIXED * M_FIXED
        tflops_f = flops_fwd / t_fwd / 1000000000000.0
        ctx.saved_dims = (qlen, k)
        ctx._time_fwd = t_fwd
        return result

    @staticmethod
    def backward(ctx, output_grad):
        input_tensor, expert_ids, weights = ctx.saved_tensors
        import random
        layer_idx = random.randint(0, 10000)
        output_grad = output_grad.contiguous().cpu()
        input_grad = torch.empty_like(input_tensor).contiguous()
        bw_start = time.time()
        ctx.cpu_infer.submit(ctx.moe.backward(output_grad.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output_grad.data_ptr(), input_grad.data_ptr()))
        ctx.cpu_infer.sync()
        bw_end = time.time()
        t_bw = bw_end - bw_start
        qlen, k = ctx.saved_dims
        flops_bw = 10 * qlen * k * H_FIXED * M_FIXED
        tflops_b = flops_bw / t_bw / 1000000000000.0
        return (input_grad.to(device=ctx.out_device), None, None, None, None, None, None)

    def unload(self):
        return

    def load_weights(self, override_key: str | None=None, device: str='cpu'):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None
        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif self.gguf_loader.has_tensor(key + '.ffn_gate_exps.weight'):
                gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
                gate_type = self.gguf_loader.get_ggml_type(key + '.ffn_gate_exps.weight')
                up_type = self.gguf_loader.get_ggml_type(key + '.ffn_up_exps.weight')
                down_type = self.gguf_loader.get_ggml_type(key + '.ffn_down_exps.weight')
            elif key + '.ffn_gate_exps.weight' in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
                gate_type = self.gguf_loader.tensor_info[key + '.ffn_gate_exps.weight']['ggml_type']
                up_type = self.gguf_loader.tensor_info[key + '.ffn_up_exps.weight']['ggml_type']
                down_type = self.gguf_loader.tensor_info[key + '.ffn_down_exps.weight']['ggml_type']
            elif key + '.ffn_down.0.weight' in self.gguf_loader.tensor_info:
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f'{key}.ffn_gate.{i}.weight')
                    up_it = self.gguf_loader.get_mmap_tensor(f'{key}.ffn_up.{i}.weight')
                    down_it = self.gguf_loader.get_mmap_tensor(f'{key}.ffn_down.{i}.weight')
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.get_ggml_type(key + '.ffn_gate.0.weight')
                up_type = self.gguf_loader.get_ggml_type(key + '.ffn_up.0.weight')
                down_type = self.gguf_loader.get_ggml_type(key + '.ffn_down.0.weight')
            else:
                raise ValueError(f'Experts {key} not found in gguf_loader')
            res = {key: {'gate': gate, 'up': up, 'down': down, 'gate_type': gate_type, 'up_type': up_type, 'down_type': down_type}}
        return res

class JOYExpertsMarlin(JOYExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, n_routed_experts: int, orig_module: nn.Module=None, device: str='cuda', **kwargs):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != 'cpu', 'Marlin experts can only be loaded on GPU'
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size
        self.up_projs = [JOYLinearMarlin(key + '.' + 'ffn_up_exps', gguf_loader, config, device=device) for i in range(self.expert_num)]
        self.gate_projs = [JOYLinearMarlin(key + '.' + 'ffn_gate_exps', gguf_loader, config, device=device) for i in range(self.expert_num)]
        self.down_projs = [JOYLinearMarlin(key + '.' + 'ffn_down_exps', gguf_loader, config, device=device) for i in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str | None=None, warmup: bool=False):
        if device is None:
            device = self.device
        assert device.lower() != 'cpu', 'Marlin experts can only be loaded on GPU'
        if w is None:
            w = self.load_weights()
            load_by_experts = True
        if load_by_experts:
            if isinstance(w, dict):
                self.gate = w['gate']
                self.up = w['up']
                self.down = w['down']
                for i in tqdm(range(self.expert_num), desc=f'Dequanting and quanting for JOYExpertsMarlin {self.key}'):
                    up_weights = self.gguf_loader.load_expert_tensor(self.key + '.ffn_up_exps.weight', self.up, i, self.elements_per_tensor, device=self.device)
                    gate_weights = self.gguf_loader.load_expert_tensor(self.key + '.ffn_gate_exps.weight', self.gate, i, self.elements_per_tensor, device=self.device)
                    down_weights = self.gguf_loader.load_expert_tensor(self.key + '.ffn_down_exps.weight', self.down, i, self.elements_per_tensor, device=self.device)
                    self.up_projs[i].load(nn.Parameter(up_weights), device=device)
                    self.gate_projs[i].load(nn.Parameter(gate_weights), device=device)
                    self.down_projs[i].load(nn.Parameter(down_weights), device=device)
                    self.loaded_experts_idx.append(i)
        elif isinstance(w, dict):
            self.gate = w['gate']
            self.up = w['up']
            self.down = w['down']
            for i in range(self.expert_num):
                self.up_projs[i].load(nn.Parameter(self.up[i, ...]), device=device)
                self.gate_projs[i].load(nn.Parameter(self.gate[i, ...]), device=device)
                self.down_projs[i].load(nn.Parameter(self.down[i, ...]), device=device)
                self.loaded_experts_idx.append(i)
        return

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None=None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        gate = None
        up = None
        down = None
        for key in keys:
            if self.gguf_loader.has_tensor(key + '.ffn_gate_exps.weight'):
                gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
            res = {'gate': gate, 'up': up, 'down': down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        org_dtype = hidden_states_cpu.dtype
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device).to(org_dtype)
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()
        final_hidden_states = torch.zeros((batch_sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)
        for expert_idx in range(self.expert_num):
            if not expert_mask[expert_idx].any():
                continue
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = self.gate_projs[expert_idx].forward(current_state)
            A = self.act_fn(G)
            U = self.up_projs[expert_idx].forward(current_state)
            H = A * U
            current_hidden_states = self.down_projs[expert_idx].forward(H) * routing_weights_cpu[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        return final_hidden_states.to(dtype=org_dtype, device=org_device)

class JOYExpertsTorch(JOYExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, n_routed_experts: int, orig_module: nn.Module=None, device: str='cpu', **kwargs):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size
        self.gate = [None for _ in range(self.expert_num)]
        self.up = [None for _ in range(self.expert_num)]
        self.down = [None for _ in range(self.expert_num)]
        self.dtype = torch.get_default_dtype()
        self.call_count = 0
        self.flops_per_call = []
        self.times = []
        self.expert_flops_details = []
        self.total_flops = 0
        h = self.config.hidden_size
        m = self.config.moe_intermediate_size
        self.params_per_expert = 3 * h * m
        self.total_params = self.expert_num * self.params_per_expert

    def load(self, w: dict | nn.Parameter | tuple | None=None, device: str | None=None, warmup: bool=False):
        if device is None:
            device = self.device
        if w is None:
            w = self.load_weights()
            load_by_experts = True
        if load_by_experts:
            if isinstance(w, dict):
                if isinstance(self.gguf_loader, SafeTensorLoader):
                    for i in tqdm(range(self.expert_num), desc=f'Loading experts(safetensors) for {self.key}'):
                        up_k = f'{self.key}.{i}.up_proj.weight'
                        gate_k = f'{self.key}.{i}.gate_proj.weight'
                        down_k = f'{self.key}.{i}.down_proj.weight'
                        self.up[i] = self.gguf_loader.load_tensor(up_k, device=self.device).contiguous()
                        self.gate[i] = self.gguf_loader.load_tensor(gate_k, device=self.device).contiguous()
                        self.down[i] = self.gguf_loader.load_tensor(down_k, device=self.device).contiguous()
                else:
                    for i in tqdm(range(self.expert_num), desc=f'Dequanting for JOYExpertsTorch {self.key}'):
                        up_weights = self.gguf_loader.load_expert_tensor(self.key + '.ffn_up_exps.weight', w['up'], i, self.elements_per_tensor, device=self.device)
                        gate_weights = self.gguf_loader.load_expert_tensor(self.key + '.ffn_gate_exps.weight', w['gate'], i, self.elements_per_tensor, device=self.device)
                        down_weights = self.gguf_loader.load_expert_tensor(self.key + '.ffn_down_exps.weight', w['down'], i, self.elements_per_tensor, device=self.device)
                        self.up[i] = up_weights
                        self.gate[i] = gate_weights
                        self.down[i] = down_weights
        elif isinstance(w, dict):
            for i in range(self.expert_num):
                self.gate[i] = w['gate'][i, ...].to(device=device, dtype=self.dtype)
                self.up[i] = w['up'][i, ...].to(device=device, dtype=self.dtype)
                self.down[i] = w['down'][i, ...].to(device=device, dtype=self.dtype)
        self.up = nn.Parameter(torch.stack(self.up, dim=0))
        self.gate = nn.Parameter(torch.stack(self.gate, dim=0))
        self.down = nn.Parameter(torch.stack(self.down, dim=0))
        return

    def unload(self):
        if self.gate is not None:
            self.gate = None
            self.up = None
            self.down = None

    def load_weights(self, override_key: str | None=None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        gate = None
        up = None
        down = None
        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif key + '.ffn_gate_exps.weight' in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
            else:
                import re
                match = re.match('model\\.layers\\.(\\d+)\\.mlp\\.experts(.*)', key)
                if match:
                    layer_id = match.group(1)
                    suffix = match.group(2)
                    key = f'blk.{layer_id}{suffix}'
                    if key + '.ffn_gate_exps.weight' in self.gguf_loader.tensor_info:
                        gate = self.gguf_loader.get_mmap_tensor(key + '.ffn_gate_exps.weight')
                        up = self.gguf_loader.get_mmap_tensor(key + '.ffn_up_exps.weight')
                        down = self.gguf_loader.get_mmap_tensor(key + '.ffn_down_exps.weight')
            res = {'gate': gate, 'up': up, 'down': down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        start_time = time.time()
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()
        final_hidden_states = torch.zeros((batch_sequence_length, hidden_dim), dtype=self.gate.dtype, device=hidden_states_cpu.device)
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)
        for expert_idx in range(self.expert_num):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = current_state @ self.gate[expert_idx, ...].T
            A = self.act_fn(G)
            U = current_state @ self.up[expert_idx, ...].T
            H = A * U
            current_hidden_states = H @ self.down[expert_idx, ...].T * routing_weights_cpu[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        call_flops = 0
        expert_details = []
        for expert_idx in range(self.expert_num):
            idx, top_x = torch.where(expert_mask[expert_idx])
            t_e = len(top_x)
            if t_e == 0:
                expert_details.append({'gate': 0, 'act': 0, 'up': 0, 'element': 0, 'down': 0, 'routing': 0})
                continue
            h = self.config.hidden_size
            m = self.config.moe_intermediate_size
            flops_gate = 2 * t_e * h * m
            flops_act = t_e * m
            flops_up = 2 * t_e * h * m
            flops_element = t_e * m
            flops_down = 2 * t_e * m * h
            flops_routing = t_e * h
            total_expert = sum([flops_gate, flops_act, flops_up, flops_element, flops_down, flops_routing])
            call_flops += total_expert
            expert_details.append({'gate': flops_gate, 'act': flops_act, 'up': flops_up, 'element': flops_element, 'down': flops_down, 'routing': flops_routing})
        self.call_count += 1
        self.flops_per_call.append(call_flops)
        self.total_flops += call_flops
        self.expert_flops_details.append(expert_details)
        self.times.append(time.time() - start_time)
        return final_hidden_states.to(dtype=org_dtype, device=org_device)
EXPERTS_MAP = {'JOYExpertsCPU': JOYExpertsCPU, 'JOYSFTExpertsCPU': JOYSFTExpertsCPU, 'JOYExpertsTorch': JOYExpertsTorch, 'JOYExpertsMarlin': JOYExpertsMarlin}

class JOYSFTExperts(BaseInjectedModule, JOYExpertsBase):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', prefill_op: str | None='JOYExpertsTorch', generate_device: str='cpu', generate_op: str | None='JOYExpertsCPU', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        JOYExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict=None, mode: InferenceState=None, warmup: bool=True):
        if not mode:
            mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError('mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD')

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, 'generate_experts is None'
            if type(self.generate_experts) == JOYSFTExpertsCPU:
                layer_idx = int(re.search('\\d+', self.key).group())
                return self.generate_experts.apply(input_tensor, expert_ids, weights, self.generate_experts.cpu_infer, self.generate_experts.moe, self.generate_experts.out_device, layer_idx)
            else:
                return self.generate_experts.forward(input_tensor, expert_ids, weights)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, 'prefill_experts is None'
            return self.prefill_experts.forward(input_tensor, expert_ids, weights)
        else:
            raise ValueError('load or set_inference_mode before forward')

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError('mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD')
from joyaiSFT.models.modeling_chatrhinoV0 import ChatrhinoV0MoE
from joyaiSFT.models.modeling_chatrhino import ChatrhinoMoE

class JOYChatrhinoV0MoE(BaseInjectedModule, ChatrhinoV0MoE):

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if sequence_length == 1 and hasattr(self.experts.generate_experts, 'submit_for_one_decode') and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
        if isinstance(self.experts, JOYExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    def moe_infer_simple(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
        return outs

    @maybe_no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        return final_out

class JOYChatrhinoMoE(BaseInjectedModule, ChatrhinoMoE):

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if sequence_length == 1 and hasattr(self.experts.generate_experts, 'submit_for_one_decode') and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
        if isinstance(self.experts, JOYExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    def moe_infer_simple(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
        return outs

    @maybe_no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        return final_out

class JOYChatrhinoMoEV2(BaseInjectedModule, ChatrhinoMoE):

    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if hasattr(self.experts.generate_experts, 'submit_for_one_decode') and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx)
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
        if isinstance(self.experts, JOYExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @maybe_no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @maybe_no_grad()
    def moe_infer_simple(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
        return outs

    @maybe_no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        return final_out

class JOYSFTExpertsV2(BaseInjectedModule, JOYExpertsBase):

    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, prefill_device: str='cuda', prefill_op: str | None='JOYExpertsTorch', generate_device: str='cpu', generate_op: str | None='JOYExpertsCPU', **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        JOYExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict=None, mode: InferenceState=None, warmup: bool=True):
        if not mode:
            mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError('mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD')

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx=0):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, 'generate_experts is None'
            return self.generate_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, 'prefill_experts is None'
            return self.prefill_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        else:
            raise ValueError('load or set_inference_mode before forward')

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError('mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD')