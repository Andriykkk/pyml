import unittest
import numpy as np
import torch
import torch.nn as nn
from pyml import tensor
from pyml.nn import Linear

class TestLinearLayerCPU(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.in_features = 10
        self.out_features = 5
        np.random.seed(42)
        torch.manual_seed(42)
    
    def compare_with_torch(self, pyml_layer, torch_layer, input_shape, rtol=1e-5, atol=1e-8):
        """Compare PyML and PyTorch linear layers"""
        np_input = np.random.randn(*input_shape)
        pyml_input = tensor(np_input, requires_grad=True)
        torch_input = torch.tensor(np_input, dtype=torch.float32, requires_grad=True)
        
        pyml_output = pyml_layer(pyml_input)
        torch_output = torch_layer(torch_input)
        
        self.assertTrue(
            np.allclose(pyml_output.numpy(), torch_output.detach().numpy(), rtol=rtol, atol=atol),
            "Forward pass mismatch"
        )
        
        if pyml_output.ndim == 1:
            pyml_output.backward()
            torch_output.backward()
        else:
            grad_output = np.random.randn(*pyml_output.shape)
            pyml_output.backward(tensor(grad_output))
            torch_output.backward(torch.tensor(grad_output))
        
        self.assertTrue(
            np.allclose(pyml_layer.weight.grad.numpy(), torch_layer.weight.grad.numpy(), rtol=rtol, atol=atol),
            "Weight gradient mismatch"
        )
        
        if pyml_layer.bias and torch_layer.bias is not None:
            self.assertTrue(
                np.allclose(pyml_layer.bias_param.grad.numpy(), torch_layer.bias.grad.numpy(), rtol=rtol, atol=atol),
                "Bias gradient mismatch"
            )
        
        self.assertTrue(
            np.allclose(pyml_input.grad.numpy(), torch_input.grad.numpy(), rtol=rtol, atol=atol),
            "Input gradient mismatch"
        )
    
    def test_linear_with_bias(self):
        """Test linear layer with bias"""
        pyml_linear = Linear(self.in_features, self.out_features, bias=True)
        torch_linear = nn.Linear(self.in_features, self.out_features, bias=True)
        
        with torch.no_grad():
            torch_linear.weight[:] = torch.tensor(pyml_linear.weight.numpy())
            torch_linear.bias[:] = torch.tensor(pyml_linear.bias_param.numpy())
        
        self.compare_with_torch(
            pyml_linear,
            torch_linear,
            (self.batch_size, self.in_features)
        )
    
    def test_linear_without_bias(self):
        """Test linear layer without bias"""
        pyml_linear = Linear(self.in_features, self.out_features, bias=False)
        torch_linear = nn.Linear(self.in_features, self.out_features, bias=False)
        
        with torch.no_grad():
            torch_linear.weight[:] = torch.tensor(pyml_linear.weight.numpy())
        
        self.compare_with_torch(
            pyml_linear,
            torch_linear,
            (self.batch_size, self.in_features)
        )
    
    def test_single_sample(self):
        """Test with single sample (no batch dimension)"""
        pyml_linear = Linear(self.in_features, self.out_features, bias=True)
        torch_linear = nn.Linear(self.in_features, self.out_features, bias=True)
        
        with torch.no_grad():
            torch_linear.weight[:] = torch.tensor(pyml_linear.weight.numpy())
            torch_linear.bias[:] = torch.tensor(pyml_linear.bias_param.numpy())
        
        self.compare_with_torch(
            pyml_linear,
            torch_linear,
            (self.in_features,)
        )
    
    def test_initialization(self):
        """Test weight initialization"""
        linear = Linear(100, 50)
        
        weights = linear.weight.numpy()
        mean = np.mean(weights)
        std = np.std(weights)
        
        expected_std = np.sqrt(2.0 / 100)
        self.assertAlmostEqual(mean, 0.0, delta=0.1)
        self.assertAlmostEqual(std, expected_std, delta=0.1)
        
        if linear.bias:
            bias = linear.bias_param.numpy()
            self.assertAlmostEqual(np.mean(bias), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(bias), expected_std, delta=0.1)
    
    def test_parameters(self):
        """Test parameters() method"""
        linear_with_bias = Linear(10, 5, bias=True)
        self.assertEqual(len(linear_with_bias.parameters()), 2)
        
        linear_without_bias = Linear(10, 5, bias=False)
        self.assertEqual(len(linear_without_bias.parameters()), 1)
    
    def test_zero_grad(self):
        """Test zero_grad() method"""
        linear = Linear(10, 5, bias=True)
        x = tensor(np.random.randn(3, 10), requires_grad=True)
        y = linear(x)
        y.backward(tensor(np.ones_like(y.numpy())))
        
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias_param.grad)
        
        linear.zero_grad()
        self.assertIsNone(linear.weight.grad)
        self.assertIsNone(linear.bias_param.grad)