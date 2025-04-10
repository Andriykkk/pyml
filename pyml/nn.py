from pyml.device import DeviceManager
import numpy as np
from pyml.tensor import tensor
from pyml.utils import kaiming_uniform

class Linear:
    def __init__(self, in_features, out_features, bias=True, device="cpu"):
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
        
        if DeviceManager.is_available(device):
            self.device = DeviceManager.get_device(device)
        else:
            raise ValueError(f"Invalid device: {device}")
        
        self.weights = tensor(
            kaiming_uniform(in_features, out_features), 
            dtype="float32",
            device=device,
            requires_grad=True
        )

        if bias:
            self.bias_param = tensor(
                np.zeros(out_features, 1).reshape(-1), 
                dtype="float32",
                device=device,
                requires_grad=True
            )
        else:
            self.bias_param = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """Forward pass of the linear layer."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False

        output = x @ self.weights.transpose()

        if self.bias:
            output = output + self.bias_param

        if squeeze_output:
            output = output.reshape(-1)

        return output
    
    def parameters(self):
        if self.bias:
            return [self.weights, self.bias_param]
        else:
            return [self.weights]
        
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}, device='{self.device}')"