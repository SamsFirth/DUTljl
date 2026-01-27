import ctypes
from contextlib import contextmanager
from typing import List, Optional, Union
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import server.envs as envs
from server.inference.distributed.cuda_wrapper import CudaRTLibrary
from server.inference.distributed.custom_all_reduce_utils import gpu_p2p_access_check
from server.inference.distributed.parallel_state import in_the_same_node_as
from server.inference.platforms import current_platform
from server.utils import cuda_device_count_stateless
import vLLMCustomAllreduce
try:
    vLLMCustomAllreduce.meta_size()
    custom_ar = True
except Exception:
    custom_ar = False

def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            print("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True

def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or inp.storage().nbytes() - inp.storage_offset() * inp.element_size() == inp.numel() * inp.element_size()

class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device], max_size=8192 * 1024) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        if not custom_ar:
            return
        self.group = group
        assert dist.get_backend(group) != dist.Backend.NCCL, 'CustomAllreduce should be attached to a non-NCCL group.'
        if not all(in_the_same_node_as(group, source_rank=0)):
            print('Custom allreduce is disabled because this process group spans across nodes.')
            return
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            return
        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            print('Custom allreduce is disabled due to an unsupported world size: %d. Supported world sizes: %s. To silence this warning, specify disable_custom_all_reduce=True explicitly.', world_size, str(CustomAllreduce._SUPPORTED_WORLD_SIZES))
            return
        if isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(',')))
        else:
            device_ids = list(range(cuda_device_count_stateless()))
        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device='cpu')
        gather_list = [torch.tensor([0], dtype=torch.int, device='cpu') for _ in range(world_size)]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]
        assert current_platform.is_cuda()
        from server.inference.platforms.cuda import CudaPlatform
        cuda_platform: CudaPlatform = current_platform
        full_nvlink = cuda_platform.is_full_nvlink(physical_device_ids)
        if world_size > 2 and (not full_nvlink):
            print("Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs. To silence this warning, specify disable_custom_all_reduce=True explicitly.")
            return
        if not _can_p2p(rank, world_size):
            print('Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.')
            return
        self.disabled = False
        self.meta_ptrs = self.create_shared_buffer(vLLMCustomAllreduce.meta_size() + max_size, group=group)
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        self.rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink
        self._ptr = vLLMCustomAllreduce.init_custom_ar(self.meta_ptrs, self.rank_data, rank, self.full_nvlink)
        vLLMCustomAllreduce.register_buffer(self._ptr, self.buffer_ptrs)

    @staticmethod
    def create_shared_buffer(size_in_bytes: int, group: Optional[ProcessGroup]=None) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)
        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)
        return pointers

    @staticmethod
    def free_shared_buffer(pointers: List[int], group: Optional[ProcessGroup]=None) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = vLLMCustomAllreduce.get_graph_buffer_ipc_meta(self._ptr)
        print('Registering %d cuda graph addresses', len(offset))
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=rank, group=self.group, device='cpu')
        handles = [d[0] for d in all_data]
        offsets = [d[1] for d in all_data]
        vLLMCustomAllreduce.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.world_size == 2 or self.full_nvlink:
            return inp_size < self.max_size
        return False

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor=None, bsz_tensor: torch.Tensor=None, registered: bool=False, is_compute_bound=False, overlap=False):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if is_compute_bound:
            sms = 2 if overlap else 36
        else:
            sms = 20 if overlap else 36
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            vLLMCustomAllreduce.all_reduce(self._ptr, inp, out, 0, 0, bsz_tensor, block_limit=sms)
        else:
            vLLMCustomAllreduce.all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size, bsz_tensor, block_limit=sms)
        return out

    def custom_all_reduce(self, input: torch.Tensor, bsz_tensor: torch.Tensor, is_compute_bound=False, overlap=False) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, bsz_tensor=bsz_tensor, registered=True, is_compute_bound=is_compute_bound, overlap=overlap)
            else:
                return torch.empty_like(input)
        else:
            return self.all_reduce(input, bsz_tensor=bsz_tensor, registered=False, is_compute_bound=is_compute_bound, overlap=overlap)

    def close(self):
        if not self.disabled and self._ptr:
            vLLMCustomAllreduce.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs)
            self.free_shared_buffer(self.buffer_ptrs)

    def __del__(self):
        self.close()