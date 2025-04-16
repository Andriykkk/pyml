import numpy as np
from pyml.tensor import tensor

_ops_softmax = {
    'cpu': {
        'forward': lambda x, dim: _softmax_forward_cpu(x, dim),
        'backward': lambda ctx, grad_output: _softmax_backward_cpu(ctx, grad_output)
    }
}

def Softmax(x=None, dim=None):
    if x is not None:
        return SoftmaxFunction.apply(x, dim)
    
    class _Softmax:
        def __init__(self, dim=None):
            self.dim = dim
        
        def __call__(self, x):
            return SoftmaxFunction.apply(x, self.dim)
    
    return _Softmax(dim=dim)

class SoftmaxFunction:
    """Function class for Softmax that handles forward/backward"""
    @staticmethod
    def forward(input_tensor, dim):
        ops = _ops_softmax[input_tensor.device.type]
        return ops['forward'](input_tensor, dim)

    @staticmethod
    def backward(ctx, grad_output):
        ops = _ops_softmax[ctx[0].device.type]
        ops['backward'](ctx, grad_output)

    @staticmethod
    def apply(input_tensor, dim=None):
        if not isinstance(input_tensor, tensor):
            input_tensor = tensor(input_tensor)
        
        if dim is None:
            dim = 0 if input_tensor.ndim == 1 else 1
        
        result = SoftmaxFunction.forward(input_tensor, dim)

        if input_tensor.requires_grad:
            result._ctx = (input_tensor, dim)
            result._grad_fn = SoftmaxFunction.backward
        
        return result

def _softmax_forward_cpu(input_tensor, dim):
    shifted = input_tensor._data - np.max(input_tensor._data, axis=dim, keepdims=True)
    exp = np.exp(shifted)
    result_data = exp / np.sum(exp, axis=dim, keepdims=True)
    
    return tensor(result_data, dtype=input_tensor.dtype,
                  device=input_tensor.device.type,
                  requires_grad=input_tensor.requires_grad)

def _softmax_backward_cpu(ctx, grad_output):
    input_tensor, dim = ctx
    s = input_tensor.softmax(dim)._data
    grad_input = grad_output._data * s - s * np.sum(grad_output._data * s, axis=dim, keepdims=True)
    
    input_tensor.backward(tensor(grad_input, device=input_tensor.device.type))

# Functional method attached to tensor class
tensor.softmax = lambda self, dim=None: SoftmaxFunction.apply(self, dim)
