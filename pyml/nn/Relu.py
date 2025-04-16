import numpy as np
from pyml.tensor import tensor

_ops_relu = {
    'cpu':{
        'forward': lambda x: _relu_forward_cpu(x),
        'backward': lambda ctx, grad_output: _relu_backward_cpu(ctx, grad_output)
    }
}

def Relu(x=None):
    if x is not None:
        return ReluFunction.apply(x)
    class _Relu:
        def __call__(self, x):
            return ReluFunction.apply(x)
    return _Relu()

class ReluFunction:
    """Function class for ReLU that handles forward/backward"""
    @staticmethod
    def forward(input_tensor):
        ops = _ops_relu[input_tensor.device.type]
        return ops['forward'](input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        ops = _ops_relu[ctx.device.type]
        ops['backward'](ctx, grad_output)

    @staticmethod
    def apply(input_tensor):
        result = ReluFunction.forward(input_tensor)
        if input_tensor.requires_grad:
            result._ctx = input_tensor
            result._grad_fn = ReluFunction.backward
        return result

def _relu_forward_cpu(input_tensor):
    result_data = np.maximum(0, input_tensor._data)
    return tensor(result_data, dtype=input_tensor.dtype, 
                device=input_tensor.device.type,
                requires_grad=input_tensor.requires_grad)

def _relu_backward_cpu(ctx, grad_output):
    mask = (ctx._data > 0).astype(float)
    grad_input = grad_output._data * mask
    
    ctx.backward(tensor(grad_input, device=ctx.device.type))

def relu(self):
    """Functional interface"""
    return ReluFunction.apply(self) 