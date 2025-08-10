"""Mock PyTorch module for testing without full PyTorch installation."""

import random
import math
from typing import Any, Optional, Tuple, List

class MockTensor:
    """Mock tensor class that mimics basic PyTorch tensor functionality."""
    
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data) if isinstance(data, tuple) else data
        else:
            self.data = [data] if not isinstance(data, list) else data
        self.dtype = dtype or 'float32'
        self._shape = self._compute_shape(self.data)
    
    def _compute_shape(self, data):
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        if not isinstance(data[0], list):
            return (len(data),)
        return (len(data), len(data[0]))
    
    def dim(self) -> int:
        return len(self._shape)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    def mean(self, dim=None):
        flat = self._flatten(self.data)
        if not flat:
            return MockTensor(0)
        return MockTensor(sum(flat) / len(flat))
    
    def _flatten(self, data):
        result = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    result.extend(self._flatten(item))
                else:
                    result.append(item)
        else:
            result.append(data)
        return result
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
        
    def item(self):
        flat = self._flatten(self.data)
        return flat[0] if flat else 0
    
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def tolist(self):
        return self.data

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
    data = input_tensor._flatten(input_tensor.data)
    sorted_data = sorted(data, reverse=True)[:k]
    indices = [data.index(val) for val in sorted_data]
    return MockTensor(sorted_data), MockTensor(indices)

def softmax(input_tensor, dim=-1):
    """Mock softmax function.""" 
    data = input_tensor._flatten(input_tensor.data)
    max_val = max(data) if data else 0
    exp_data = [math.exp(x - max_val) for x in data]
    sum_exp = sum(exp_data)
    softmax_data = [x / sum_exp for x in exp_data] if sum_exp > 0 else data
    return MockTensor(softmax_data)

def randint(low, high, size):
    """Mock randint function."""
    if isinstance(size, tuple):
        return MockTensor([random.randint(low, high-1) for _ in range(size[0])])
    return MockTensor(random.randint(low, high-1))

def cosine_similarity(x1, x2, dim=0):
    """Mock cosine similarity."""
    data1 = x1._flatten(x1.data)
    data2 = x2._flatten(x2.data)
    dot_product = sum(a * b for a, b in zip(data1, data2))
    norm1 = math.sqrt(sum(x * x for x in data1))
    norm2 = math.sqrt(sum(x * x for x in data2))
    return MockTensor(dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0)

def cat(tensors, dim=0):
    """Mock concatenation."""
    all_data = []
    for t in tensors:
        all_data.extend(t._flatten(t.data))
    return MockTensor(all_data)

def norm(input_tensor):
    """Mock norm function."""
    data = input_tensor._flatten(input_tensor.data)
    return MockTensor(math.sqrt(sum(x * x for x in data)))

def log(input_tensor):
    """Mock log function."""
    data = input_tensor._flatten(input_tensor.data)
    return MockTensor([math.log(x + 1e-10) for x in data])

def sum(input_tensor, dim=None):
    """Mock sum function."""
    data = input_tensor._flatten(input_tensor.data)
    return MockTensor(sum(data))

def rand(size):
    """Mock random tensor generation."""
    if isinstance(size, int):
        size = (size,)
    import random
    flat_size = 1
    for s in size:
        flat_size *= s
    return MockTensor([random.random() for _ in range(flat_size)])

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
    
    class Linear(MockModule):
        """Mock linear layer."""
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = MockTensor([[0.1] * in_features for _ in range(out_features)])
            if bias:
                self.bias = MockTensor([0.0] * out_features)
            else:
                self.bias = None

# Create mock torch module structure
class MockTorch:
    """Mock torch module."""
    
    tensor = tensor
    Tensor = MockTensor  # Add Tensor class reference
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