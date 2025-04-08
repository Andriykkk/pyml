import numpy as np
from .device import Device, DeviceManager

class tensor:
    def __init__(self, data=None, dtype=None, device="cpu", requires_grad=False):
        """
        Create a new tensor.
        
        Args:
            data: Input data (array-like, scalar, or None)
            dtype: Data type (e.g., 'float32', 'int64')
            device: Device to store tensor ('cpu' or 'cuda')
            requires_grad: If True, tracks computation history for gradients
        """
        self.device = DeviceManager.get_device(device)
        self.requires_grad = requires_grad
        self.grad = None

        if data is None:
            self._data = np.array(0, dtype=dtype)
        elif isinstance(data, (list, tuple)) or (hasattr(data, "__array__")) and not isinstance(data, np.ndarray):
            self._data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(dtype) if dtype is not None else data.copy()
        elif isinstance(data, (int, float)):
            self._data = np.array(data, dtype=dtype)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        
    # Operations
    def __neg__(self):
        return tensor(-self._data, dtype=self.dtype, device=self.device.type)

    def __add__(self, other):
        """Element-wise addition with automatic broadcasting"""
        try:
            other_data = other._data if isinstance(other, tensor) else other
            return tensor(self._data + other_data, dtype=self.dtype, device=self.device.type)
        except Exception as e:
            raise e

    def __sub__(self, other):
        """Element-wise subtraction with automatic broadcasting"""
        try:
            other_data = other._data if isinstance(other, tensor) else other
            return tensor(self._data - other_data, dtype=self.dtype, device=self.device.type)
        except Exception as e:
            raise TypeError(f"Subtraction not supported between types {type(self)} and {type(other)}") from e

    def __mul__(self, other):
        """Element-wise multiplication with automatic broadcasting"""
        try:
            other_data = other._data if isinstance(other, tensor) else other
            return tensor(self._data * other_data, dtype=self.dtype, device=self.device.type)
        except Exception as e:
            raise TypeError(f"Multiplication not supported between types {type(self)} and {type(other)}") from e

    def __truediv__(self, other):
        """Element-wise division with automatic broadcasting"""
        try:
            other_data = other._data if isinstance(other, tensor) else other
            return tensor(self._data / other_data, dtype=self.dtype, device=self.device.type)
        except Exception as e:
            raise TypeError(f"Division not supported between types {type(self)} and {type(other)}") from e
        
    def matmul(self, other):
        """Matrix multiplication with automatic broadcasting using NumPy's implementation"""
        if not isinstance(other, (tensor, np.ndarray, list, tuple, int, float)):
            raise TypeError(f"Unsupported type for matmul: {type(other)}")
        
        other_data = other._data if isinstance(other, tensor) else other
        
        try:
            return tensor(np.matmul(self._data, other_data), 
                        dtype=self.dtype, 
                        device=self.device.type)
        except ValueError as e:
            if isinstance(other, tensor):
                raise ValueError(
                    f"matmul shape mismatch: {self.shape} and {other.shape}. "
                    f"Matrix multiplication requires last dim of first tensor ({self.shape[-1]}) "
                    f"to match second-to-last dim of second tensor ({other.shape[-2] if other.ndim >= 2 else other.shape[0]})"
                ) from e
            raise

    def __matmul__(self, other):
        """Support for @ operator"""
        return self.matmul(other)
    
    def transpose(self, *axes):
        """Transpose the tensor"""
        return tensor(self._data.transpose(*axes), dtype=self.dtype, device=self.device.type)

    # Properties
    @property
    def dtype(self):
        """Get the data type of the tensor"""
        return str(self._data.dtype)
    
    @property
    def shape(self):
        """Get the shape of the tensor"""
        return self._data.shape
    
    @property
    def ndim(self):
        """Get the number of dimensions of the tensor"""
        return self._data.ndim
    
    @property
    def size(self):
        """Get the total number of elements in the tensor"""
        return self._data.size
    
    # Magic methods
    def numpy(self):
        """Convert tensor to numpy array"""
        return self._data.copy()
    
    def __repr__(self):
        return f"tensor({self._data}, dtype={self.dtype}, device='{self.device}')"
    
    def __str__(self):
        return str(self._data)
    
    def __array__(self):
        return self._data
    
    def __getitem__(self, index):
        """Handle indexing and slicing."""
        if isinstance(index, tuple):  
            return tensor(self._data[index], dtype=self.dtype, device=self.device.type, requires_grad=self.requires_grad)
        else: 
            return tensor(self._data[index], dtype=self.dtype, device=self.device.type, requires_grad=self.requires_grad)

    # Class methods
    @classmethod
    def zeros(cls, *size, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor filled with zeros"""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        return cls(np.zeros(size, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, *size, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor filled with ones"""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        return cls(np.ones(size, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, *size, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor with random values in [0, 1)"""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        return cls(np.random.rand(*size), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, *size, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor with random values from standard normal distribution"""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        return cls(np.random.randn(*size), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def randint(cls, low, high=None, size=None, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor with random integers"""
        if dtype and not np.issubdtype(dtype, np.integer):
            raise ValueError("dtype must be an integer type for randint")
    
        if dtype is None:
            dtype = np.int64
        return cls(np.random.randint(low, high, size, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def empty(cls, *size, dtype=None, device="cpu", requires_grad=False):
        """Create an uninitialized tensor"""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        return cls(np.empty(size, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def full(cls, size, fill_value, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor filled with a specific value"""
        return cls(np.full(size, fill_value, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def arange(cls, start, stop=None, step=1, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor with a range of values"""
        return cls(np.arange(start, stop, step, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def eye(cls, n, m=None, dtype=None, device="cpu", requires_grad=False):
        """Create a 2D tensor with ones on the diagonal and zeros elsewhere"""
        return cls(np.eye(n, m, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def linspace(cls, start, stop, num=50, dtype=None, device="cpu", requires_grad=False):
        """Create a tensor with evenly spaced numbers over a specified interval"""
        return cls(np.linspace(start, stop, num, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def from_numpy(cls, ndarray):
        """Create a tensor from a numpy array"""
        return cls(ndarray)