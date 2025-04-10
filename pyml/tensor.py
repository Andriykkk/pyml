import numpy as np
from .device import DeviceManager

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
        self._grad_fn = None
        self._ctx = None

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
        
    def backward(self, grad=None):
        """
        Compute gradients via backpropagation.
        
        Args:
            grad: Gradient of the loss with respect to this tensor (default is 1 for scalar tensors)
        """
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.shape != ():
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = tensor(1.0, device=self.device.type)
        elif isinstance(grad, (int, float)):
            grad = tensor(grad, device=self.device.type)
        
        if self.grad is None:
            self.grad = grad if isinstance(grad, tensor) else tensor(grad)
        else:
            self.grad = self.grad + grad
        
        if self._ctx is not None:
            self._grad_fn(self._ctx, grad)

    def zero_grad(self):
        """Reset gradients to zero"""
        self.grad = None
        
    # Operations
    def __neg__(self):
        result = tensor(-self._data, dtype=self.dtype, device=self.device.type)

        if result.requires_grad:
            result._ctx = (self)
            result._grad_fn = _neg_backward

        return result

    def __add__(self, other):
        """Element-wise addition with automatic broadcasting"""
        other = other if isinstance(other, tensor) else tensor(other, device=self.device.type)

        if self.device.type != other.device.type:
            raise RuntimeError("Tensors must be on the same device")

        result_data = self._data + other._data
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                    requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, other)
            result._grad_fn = _add_backward
        
        return result

    def __sub__(self, other):
        """Element-wise subtraction with automatic broadcasting"""
        other = other if isinstance(other, tensor) else tensor(other, device=self.device.type)
        
        if self.device.type != other.device.type:
            raise RuntimeError("Tensors must be on the same device")
        
        result_data = self._data - other._data
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, other)
            result._grad_fn = _sub_backward
        
        return result

    def __mul__(self, other):
        """Element-wise multiplication with automatic broadcasting"""
        other = other if isinstance(other, tensor) else tensor(other, device=self.device.type)
        
        if self.device.type != other.device.type:
            raise RuntimeError("Tensors must be on the same device")
        
        result_data = self._data * other._data
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, other)
            result._grad_fn = _mul_backward
        
        return result

    def __truediv__(self, other):
        """Element-wise division with automatic broadcasting"""
        other = other if isinstance(other, tensor) else tensor(other, device=self.device.type)

        if self.device.type != other.device.type:
            raise RuntimeError("Tensors must be on the same device")
        
        result_data = self._data / other._data
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, other)
            result._grad_fn = _div_backward
        
        return result
        
    def matmul(self, other):
        """Matrix multiplication with autograd support"""
        if not isinstance(other, tensor):
            raise TypeError(f"Unsupported type for matmul: {type(other)}")
        
        result_data = np.matmul(self._data, other._data)
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, other)
            result._grad_fn = _matmul_backward
        
        return result

    def __matmul__(self, other):
        """Support for @ operator"""
        if self.device.type != other.device.type:
            raise RuntimeError("Tensors must be on the same device")
        
        return self.matmul(other)
    
    def transpose(self, *axes):
            """Transpose the tensor dimensions with autograd support"""
            if len(axes) == 0:
                transposed_data = np.transpose(self._data)
            else:
                transposed_data = np.transpose(self._data, axes)

            result = tensor(transposed_data, dtype=self.dtype, device=self.device.type,
                    requires_grad=self.requires_grad)
            
            if result.requires_grad:
                result._ctx = (self, axes if len(axes) > 0 else None)
                result._grad_fn = _transpose_backward
            
            return result
    
    def sum(self, axis=None, keepdims=False, **kwargs):
        """Sum of tensor elements with autograd support"""
        result_data = np.sum(self._data, axis=axis, keepdims=keepdims, **kwargs)
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                       requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, axis, keepdims)
            result._grad_fn = _sum_backward
        
        return result
    
    def max(self, axis=None, keepdims=False):
        """Compute the maximum of tensor elements along given axis"""
        result_data = np.max(self._data, axis=axis, keepdims=keepdims)
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                      requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, axis, keepdims)
            result._grad_fn = _max_backward
        
        return result
    
    def mean(self, axis=None, keepdims=False):
        """Mean of tensor elements with autograd support"""
        result_data = np.mean(self._data, axis=axis, keepdims=keepdims)
        
        result = tensor(result_data, dtype=self.dtype, device=self.device.type,
                       requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._ctx = (self, axis, keepdims)
            result._grad_fn = _mean_backward
        
        return result

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
    
def _neg_backward(grad):
    raise NotImplementedError

def _add_backward(ctx, grad):
    a, b = ctx
    a, b = ctx
    if a.requires_grad:
        a_grad = grad
        if a.shape != grad.shape:
            a_grad = _sum_to_shape(grad, a.shape)
        a.backward(a_grad)
    if b.requires_grad:
        b_grad = grad
        if b.shape != grad.shape:
            b_grad = _sum_to_shape(grad, b.shape)
        b.backward(b_grad)

def _sub_backward(ctx, grad):
    a, b = ctx
    a, b = ctx
    if a.requires_grad:
        a_grad = grad
        if a.shape != grad.shape:
            a_grad = _sum_to_shape(grad, a.shape)
        a.backward(a_grad)
    if b.requires_grad:
        b_grad = -grad
        if b.shape != grad.shape:
            b_grad = _sum_to_shape(grad, b.shape)
        b.backward(b_grad)

def _mul_backward(ctx, grad):
    a, b = ctx
    if a.requires_grad:
        grad_a = grad * b._data
        if a.shape != grad_a.shape:
            grad_a = _sum_to_shape(grad_a, a.shape)
        a.backward(grad_a)
    if b.requires_grad:
        grad_b = grad * a._data
        if b.shape != grad_b.shape:
            grad_b = _sum_to_shape(grad_b, b.shape)
        b.backward(grad_b)

def _div_backward(ctx, grad):
    a, b = ctx
    if a.requires_grad:
        grad_a = grad / b._data
        if a.shape != grad_a.shape:
            grad_a = _sum_to_shape(grad_a, a.shape)
        a.backward(grad_a)
    if b.requires_grad:
        grad_b = -grad * a._data / (b._data ** 2)
        if b.shape != grad_b.shape:
            grad_b = _sum_to_shape(grad_b, b.shape)
        b.backward(grad_b)

def _matmul_backward(ctx, grad):
    a, b = ctx
    if a.requires_grad:
        if grad._data.ndim == 1 and b._data.ndim == 1:
            grad_a = np.outer(grad._data, b._data)
        elif grad._data.ndim == 2 and b._data.ndim == 2:
            grad_a = np.matmul(grad._data, b._data.T)
        else:
            grad_a = np.matmul(grad._data, np.swapaxes(b._data, -1, -2))
        a.backward(tensor(grad_a, device=a.device.type))
    
    if b.requires_grad:
        if a._data.ndim == 1 and grad._data.ndim == 1:
            grad_b = np.outer(a._data, grad._data)
        elif a._data.ndim == 2 and grad._data.ndim == 2:
            grad_b = np.matmul(a._data.T, grad._data)
        else:
            grad_b = np.matmul(np.swapaxes(a._data, -1, -2), grad._data)
        b.backward(tensor(grad_b, device=b.device.type))

def _transpose_backward(ctx, grad):
    a, axes = ctx
    if a.requires_grad:
        if axes is None:
            grad_a = np.transpose(grad._data)
        else:
            original_axes = list(range(len(axes)))
            for i, axis in enumerate(axes):
                original_axes[axis] = i
            grad_a = np.transpose(grad._data, original_axes)
        a.backward(tensor(grad_a, device=a.device.type))

def _sum_backward(ctx, grad):
    a, axis, keepdims = ctx
    if a.requires_grad:
        if axis is None:
            grad_a = np.ones_like(a._data) * grad._data
        else:
            grad_a = np.zeros_like(a._data)
            if keepdims:
                expanded_grad = grad._data
            else:
                expanded_shape = list(a.shape)
                if isinstance(axis, int):
                    expanded_shape[axis] = 1
                else:
                    for ax in sorted(axis):
                        expanded_shape[ax] = 1
                expanded_grad = np.reshape(grad._data, expanded_shape)
            
            grad_a = np.broadcast_to(expanded_grad, a.shape)
        a.backward(tensor(grad_a, device=a.device.type))

def _mean_backward(ctx, grad):
    a, axis, keepdims = ctx
    if a.requires_grad:
        if axis is None:
            count = a.size
        elif isinstance(axis, int):
            count = a.shape[axis]
        else:
            count = np.prod([a.shape[ax] for ax in axis])
        
        scaled_grad = grad._data / count
        
        if axis is None:
            grad_a = np.ones_like(a._data) * scaled_grad
        else:
            grad_a = np.zeros_like(a._data)
            if keepdims:
                expanded_grad = scaled_grad
            else:
                expanded_shape = list(a.shape)
                if isinstance(axis, int):
                    expanded_shape[axis] = 1
                else:
                    for ax in sorted(axis):
                        expanded_shape[ax] = 1
                expanded_grad = np.reshape(scaled_grad, expanded_shape)
            
            grad_a = np.broadcast_to(expanded_grad, a.shape)
        a.backward(tensor(grad_a, device=a.device.type))

def _max_backward(ctx, grad):
    """
    Backward pass for the max function.
    
    The gradient flows only to the elements that were selected in the forward pass.
    All other elements receive zero gradient.
    """
    input_tensor, axis, keepdims = ctx

    if input_tensor.requires_grad:
        input_data = input_tensor._data

        if axis is None:
            max_values = np.max(input_data)
            mask = (input_data == max_values)
        else:
            max_values = np.max(input_data, axis=axis, keepdims=True)
            mask = (input_data == max_values)

            mask = mask.astype(np.float32)
            counts = np.sum(mask, axis=axis, keepdims=True)
            mask = mask / counts

        if not keepdims and axis is not None:
            if isinstance(axis, int):
                axis = (axis,)

            expanded_shape = list(input_tensor.shape)
            for ax in sorted(axis):
                expanded_shape[ax] = 1

            grad_data = np.reshape(grad._data, expanded_shape)
        else:
            grad_data = grad._data

        grad_data = grad_data * mask

        input_tensor.backward(tensor(grad_data, device=input_tensor.device.type))


def _sum_to_shape(grad, shape):
    """Sum gradients to match the original tensor shape (for broadcasting)"""
    sum_axes = tuple(range(len(grad.shape) - len(shape)))

    for i, dim in enumerate(shape):
        if dim == 1:
            sum_axes += (i,)

    if sum_axes:
        return np.sum(grad, axis=sum_axes, keepdims=True)
    return grad
