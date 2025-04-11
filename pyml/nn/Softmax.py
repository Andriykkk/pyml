import numpy as np
from pyml.tensor import tensor

_ops_softmax = {
    'cpu': {
        'forward': lambda x, dim: _softmax_forward_cpu(x, dim),
        'backward': lambda ctx, grad_output: _softmax_backward_cpu(ctx, grad_output)
    }
}

class Softmax:
    """Softmax activation function that can be called directly"""
    def __init__(self, dim=None):
        self.dim = dim
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """Forward pass of softmax"""
        if not isinstance(x, tensor):
            x = tensor(x)
            
        if self.dim is None:
            dim = 0 if x.ndim == 1 else 1
        else:
            dim = self.dim
            
        result = _ops_softmax[x.device.type]['forward'](x, dim)
        
        if result.requires_grad:
            result._ctx = (x, dim)
            result._grad_fn = _ops_softmax[x.device.type]['backward']
            
        return result

def softmax(x, dim=None):
    """Functional interface for softmax"""
    return Softmax(dim=dim)(x)



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

tensor.softmax = lambda self, dim=None: softmax(self, dim) 