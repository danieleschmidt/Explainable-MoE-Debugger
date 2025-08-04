"""Mock PyTorch module for testing without full PyTorch installation."""

import numpy as np
from typing import Any, Optional, Tuple, List

class MockTensor:
    """Mock tensor class that mimics basic PyTorch tensor functionality."""
    
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array([data])
        self.dtype = dtype or 'float32'
    
    def dim(self) -> int:
        return len(self.data.shape)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    def mean(self, dim=None):
        if dim is None:
            return MockTensor(np.mean(self.data))
        return MockTensor(np.mean(self.data, axis=dim))
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
    
    def item(self):
        return self.data.item()
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def tolist(self):
        return self.data.tolist()

class MockModule:
    """Mock PyTorch module class."""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self):
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            for sub_name, param in module.named_parameters():
                yield f"{name}.{sub_name}", param
    
    def named_modules(self):
        yield "", self
        for name, module in self._modules.items():
            yield name, module
            for sub_name, sub_module in module.named_modules():
                if sub_name:  # Skip empty names
                    yield f"{name}.{sub_name}", sub_module

def tensor(data, dtype=None):
    """Create a mock tensor."""
    return MockTensor(data, dtype)

def topk(input_tensor, k, dim=-1):
    """Mock topk function."""
    data = input_tensor.data
    if dim == -1:
        dim = len(data.shape) - 1
    
    indices = np.argsort(data, axis=dim)[..., -k:]
    values = np.sort(data, axis=dim)[..., -k:]
    
    return MockTensor(values), MockTensor(indices)

def softmax(input_tensor, dim=-1):
    """Mock softmax function."""
    data = input_tensor.data
    exp_data = np.exp(data - np.max(data, axis=dim, keepdims=True))
    return MockTensor(exp_data / np.sum(exp_data, axis=dim, keepdims=True))

def cosine_similarity(x1, x2, dim=0):
    """Mock cosine similarity."""
    dot_product = np.dot(x1.data, x2.data)
    norm1 = np.linalg.norm(x1.data)
    norm2 = np.linalg.norm(x2.data)
    return MockTensor(dot_product / (norm1 * norm2))

def cat(tensors, dim=0):
    """Mock concatenation."""
    arrays = [t.data for t in tensors]
    return MockTensor(np.concatenate(arrays, axis=dim))

def norm(input_tensor):
    """Mock norm function."""
    return MockTensor(np.linalg.norm(input_tensor.data))

def log(input_tensor):
    """Mock log function."""
    return MockTensor(np.log(input_tensor.data + 1e-10))

def sum(input_tensor, dim=None):
    """Mock sum function."""
    if dim is None:
        return MockTensor(np.sum(input_tensor.data))
    return MockTensor(np.sum(input_tensor.data, axis=dim))

def rand(size):
    """Mock random tensor generation."""
    if isinstance(size, int):
        size = (size,)
    return MockTensor(np.random.rand(*size))

class cuda:
    """Mock CUDA module."""
    
    @staticmethod
    def is_available():
        return False
    
    @staticmethod
    def memory_allocated():
        return 0
    
    @staticmethod
    def reset_peak_memory_stats():
        pass

class nn:
    """Mock neural network module."""
    
    class Module(MockModule):
        pass

# Create mock torch module structure
class MockTorch:
    """Mock torch module."""
    
    tensor = tensor
    topk = topk
    softmax = softmax
    cosine_similarity = cosine_similarity
    cat = cat
    norm = norm
    log = log
    sum = sum
    rand = rand
    cuda = cuda
    nn = nn
    
    @staticmethod
    def load(*args, **kwargs):
        return MockModule()

# Make it available as torch
torch = MockTorch()