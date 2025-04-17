import numpy as np
from pyml.tensor import tensor

class Module:
    """Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can be nested in other Modules in a tree structure. You can
    assign them as regular attributes:

    >>> class Model(Module):
    >>>     def __init__(self):
    >>>         self.linear1 = Linear(10, 20)
    >>>         self.linear2 = Linear(20, 30)

    Submodules assigned as attributes will be registered and their
    parameters (if any) will be part of this Module's parameters().
    """
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, tensor) and value.requires_grad:
            self._parameters[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        """Returns iterator over module parameters"""
        for name, param in self._parameters.items():
            yield param
        for name, param in self._modules.items():
            yield from param.parameters()

    def zero_grad(self):
        """Set gradients to zero"""
        for p in self.parameters():
            if p.grad is not None:
                p.grad = None

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Args:
            input: The input to the module.

        Returns:
            The output of the module.
        """
        raise NotImplementedError
    
    def __call__(self, *input):
        return self.forward(*input)