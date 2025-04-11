import numpy as np
from pyml.tensor import tensor
from pyml.utils import kaiming_uniform

class Linear:
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        """
        A linear layer (fully connected layer) with optional bias.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If False, the layer will not learn an additive bias
            device: Device to store parameters ('cpu' or 'cuda')
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        
        # Initialize weights
        self.weight = tensor(
            kaiming_uniform(in_features, out_features),
            dtype='float32',
            device=device,
            requires_grad=True
        )
        
        # Initialize bias if requested
        if bias:
            self.bias_param = tensor(
                np.reshape(kaiming_uniform(out_features, 1), (-1,)),
                dtype='float32',
                device=device,
                requires_grad=True
            )
        else:
            self.bias_param = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """Forward pass of the linear layer using pyml.tensor operations"""
        squeeze_output = False
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True

        # Use pyml.tensor's matmul and transpose
        output = x @ self.weight.transpose()
        if self.bias:
            output = output + self.bias_param

        if squeeze_output:
            output = output.reshape(-1)

        if not isinstance(output, tensor):
            output = tensor(output, dtype=x.dtype, requires_grad=True)

        return output
    
    def parameters(self):
        """Return all trainable parameters"""
        if self.bias:
            return [self.weight, self.bias_param]
        return [self.weight]
    
    def zero_grad(self):
        """Reset gradients for all parameters"""
        for param in self.parameters():
            param.grad = None
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"