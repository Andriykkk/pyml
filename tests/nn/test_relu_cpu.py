import unittest
import numpy as np
import torch
import torch.nn as nn
from pyml import tensor
from pyml.nn import Linear
from pyml.nn import Relu

class TestReLU(unittest.TestCase):
    def setUp(self):
        self.relu_fn = Relu()
        
    def test_relu_forward(self):
        # Test basic functionality
        x = tensor([-1.0, 0.0, 2.0])
        y = self.relu_fn(x)
        self.assertTrue(np.array_equal(y.numpy(), [0.0, 0.0, 2.0]))
        
        # Test saved module
        y2 = self.relu_fn(x)
        self.assertTrue(np.array_equal(y2.numpy(), [0.0, 0.0, 2.0]))
        
    def test_relu_backward(self):
        # Test gradient computation
        x = tensor([-1.0, 0.5, 0.0, 3.0], requires_grad=True)
        y = self.relu_fn(x)
        y.backward(tensor([1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.array_equal(x.grad.numpy(), [0.0, 1.0, 0.0, 1.0]))
        
        # Test with different output gradients
        x = tensor([-2.0, 1.0], requires_grad=True)
        y = self.relu_fn(x)
        y.backward(tensor([2.0, 3.0]))
        self.assertTrue(np.array_equal(x.grad.numpy(), [0.0, 3.0]))
    
    def test_relu_device_consistency(self):
        x_cpu = tensor([-1.0, 1.0], device='cpu', requires_grad=True)
        y_cpu = self.relu_fn(x_cpu)
        y_cpu.backward(tensor([1.0, 1.0]))
        self.assertEqual(x_cpu.grad.device.type, 'cpu')  
        
        x_cpu.zero_grad()

        y2_cpu = self.relu_fn(x_cpu)
        y2_cpu.backward(tensor([1.0, 1.0]))
        self.assertTrue(np.array_equal(x_cpu.grad.numpy(), [0.0, 1.0]))
         
    def test_relu_memory(self):
        x = tensor(np.random.randn(1000, 1000), requires_grad=True)
        y = self.relu_fn(x)
        y.backward(tensor(np.ones((1000, 1000))))
        del y
        self.assertTrue(x.grad is not None)
        
    def test_relu_reference(self):
        # Test saved function works multiple times
        x = tensor([-1.0, 1.0], requires_grad=True)
        relu = Relu()
        
        y1 = relu(x)
        y1.backward(tensor([1.0, 1.0]))
        grad1 = x.grad.numpy().copy()
        x.zero_grad()
        
        y2 = relu(x)
        y2.backward(tensor([1.0, 1.0]))
        grad2 = x.grad.numpy()
        
        self.assertTrue(np.array_equal(grad1, grad2))