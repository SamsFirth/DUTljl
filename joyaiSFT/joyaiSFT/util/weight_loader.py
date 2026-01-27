from abc import ABC, abstractmethod
import os
import torch
import numpy as np
from safetensors import safe_open
from typing import Dict, Any, Optional, Union

class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    Defines the interface that all model loaders must implement.
    """

    @abstractmethod
    def load_tensor(self, name: str, device: str='cpu') -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        pass

    @classmethod
    @abstractmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if this loader supports the given path, False otherwise
        """
        pass

class SafeTensorLoader(ModelLoader):
    """
    Loader for SafeTensor format models.
    """

    def __init__(self, path: str):
        """
        Initialize the SafeTensor loader.
        
        Args:
            path: Path to the model directory or file
        """
        self.tensor_file_map = {}
        self.file_handle_map = {}
        self._load_tensor_file_map(path)

    def _load_tensor_file_map(self, path: str) -> None:
        """
        Load the tensor file map from the given path.
        
        Args:
            path: Path to the model directory or file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path not found: {path}')
        if os.path.isfile(path):
            folder_path = os.path.dirname(path)
        else:
            folder_path = path
        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith('.safetensors'):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework='pt')
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f'Error opening Safetensor file {file_path}: {e}')
                            continue
                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f'Error reading Safetensor file {file_path}: {e}')
        if not found_safetensor:
            print(f'No Safetensor files found in {folder_path}')

    def load_tensor(self, name: str, device: str='cpu') -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        if name not in self.tensor_file_map:
            raise KeyError(f'Key {name} not found in Safetensor files')
        file = self.tensor_file_map[name]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f'File {file} not found in Safetensor files')
        tensor = f.get_tensor(name)
        return tensor.to(device)

    def load_dequantized_tensor(self, name: str, device: str='cpu') -> torch.Tensor:
        """
        Load and dequantize a tensor.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The dequantized tensor
        """
        if name not in self.tensor_file_map:
            raise KeyError(f'Key {name} not found in Safetensor files')
        file = self.tensor_file_map[name]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f'File {file} not found in Safetensor files')
        tensor = f.get_tensor(name).to(device)
        if name.endswith('.weight'):
            if name[:-7] + '.weight_scale_inv' in self.tensor_file_map:
                weight_scale_inv = f.get_tensor(name[:-7] + '.weight_scale_inv').to(device)
                from joyaiSFT.joyaiSFT_ext.triton.fp8gemm import weight_dequant
                tensor = weight_dequant(tensor, weight_scale_inv)
        return tensor.to(device)

    def close_all_handles(self) -> None:
        """
        Close all file handles.
        """
        for handle in self.file_handle_map.values():
            handle.close()
        self.file_handle_map.clear()

    @classmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if safetensor files are found in the path, False otherwise
        """
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            if path.endswith('.safetensors'):
                return True
            folder_path = os.path.dirname(path)
        else:
            folder_path = path
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.safetensors'):
                    return True
        return False

class GGUFLoader(ModelLoader):
    """
    Loader for GGUF format models.
    """

    def __init__(self, path: str):
        """
        Initialize the GGUF loader.
        
        Args:
            path: Path to the model directory or file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'GGUF dir not found: {path}')
        if os.path.isfile(path):
            self.gguf_path = os.path.dirname(path)
        else:
            self.gguf_path = path
        self.tensor_info = {}
        self.tensor_file_map = {}
        self.file_data_map = {}
        self.gguf_file_meta = {}
        self.safetensor_loader = None
        found_gguf = False
        for root, _, files in os.walk(self.gguf_path):
            for file in files:
                if file.endswith('.gguf'):
                    found_gguf = True
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        self._load_gguf(f)
                        if file_path not in self.file_data_map:
                            self.file_data_map[file_path] = np.memmap(file_path, mode='r')
        if not found_gguf:
            raise FileNotFoundError(f'Cannot find any .gguf files in: {self.gguf_path}')

    def _load_gguf(self, f) -> None:
        """
        Load GGUF file metadata and tensor info.
        
        Args:
            f: File handle of the GGUF file
        """
        f.seek(0)
        assert f.read(4) == b'GGUF'
        values = struct.unpack('<IQQ', f.read(4 + 8 + 8))
        version, n_tensors, n_kv = values
        if version != 3:
            warnings.warn(f'Version {version} has never been tested, might not work')
        info = {}
        for _ in range(n_kv):
            name = self._read_value(f, 8)
            data_type = struct.unpack('<I', f.read(4))[0]
            info[name] = self._read_value(f, data_type)
        tensor_info = {}
        for _ in range(n_tensors):
            name = self._read_value(f, 8)
            shape_len = self._read_value(f, 4)
            shape = [self._read_value(f, 10) for _ in range(shape_len)]
            ggml_type = self._read_value(f, 4)
            offset = self._read_value(f, 10)
            tensor_info[name] = {'ggml_type': ggml_type, 'shape': shape, 'offset': offset}
        start = f.tell()
        alignment = info.get('general.alignment', 32)
        for t in tensor_info.values():
            offset = start + t['offset']
            offset += (alignment - offset % alignment) % alignment
            t['offset'] = offset
        for name in tensor_info:
            self.tensor_file_map[name] = f.name
        self.tensor_info.update(tensor_info)
        self.gguf_file_meta.update(info)

    def _read_value(self, f, data_type) -> Any:
        """
        Read a value from the file according to its data type.
        
        Args:
            f: File handle
            data_type: Type of data to read
            
        Returns:
            The read value
        """
        if data_type == 8:
            length = struct.unpack('<Q', f.read(8))[0]
            return f.read(length).decode('utf-8')
        elif data_type == 4:
            return struct.unpack('<I', f.read(4))[0]
        elif data_type == 10:
            return struct.unpack('<Q', f.read(8))[0]
        return None

    def load_tensor(self, name: str, device: str='cpu') -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        return self.load_gguf_tensor(name, device)

    def load_gguf_tensor(self, name: str, device: str='cpu', target_dtype=None) -> torch.Tensor:
        """
        Load a GGUF tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            target_dtype: Target data type for the tensor
            
        Returns:
            The loaded tensor
        """
        if name not in self.tensor_info:
            raise KeyError(f'Tensor {name} not found')
        return torch.zeros(1, device=device)

    @classmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if GGUF files are found in the path, False otherwise
        """
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            return path.endswith('.gguf')
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.gguf'):
                    return True
        return False