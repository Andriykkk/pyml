import unittest
import numpy as np
import torch
import torch.nn as nn
from pyml.tensor import tensor
from pyml.nn import Linear, Relu, Softmax
from pyml.nn.module import Module  

class TestModule(unittest.TestCase):
    def test_module_parameter_registration(self):
        class TestNet(Module):
            def __init__(self):
                super().__init__()
                self.linear1 = Linear(10, 20)
                self.linear2 = Linear(20, 30, bias=False)
                self.bias_param = tensor(np.ones(5), requires_grad=True)
                self.non_param_attr = np.array([1, 2, 3])

            def forward(self, x):
                return x

        net = TestNet()
        params = list(net.parameters())
        self.assertEqual(len(params), 4)
        self.assertTrue(net.linear1.weight in params)
        self.assertTrue(net.linear1.bias in params)
        self.assertTrue(net.linear2.weight in params)
        self.assertTrue(net.bias_param in params)
        self.assertTrue(hasattr(net, 'non_param_attr')) 

    def test_module_nested_parameter_registration(self):
        class InnerNet(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        class OuterNet(Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerNet()
                self.linear_out = Linear(10, 2)
                self.non_param = [1, 2]

            def forward(self, x):
                return self.linear_out(self.inner(x))

        net = OuterNet()
        params = list(net.parameters())
        self.assertEqual(len(params), 4)
        self.assertTrue(net.inner.linear.weight in params)
        self.assertTrue(net.inner.linear.bias in params)
        self.assertTrue(net.linear_out.weight in params)
        self.assertTrue(net.linear_out.bias in params)
        self.assertFalse(net.non_param in params)
        self.assertTrue(hasattr(net, 'non_param'))

    def test_module_zero_grad(self):
        class TestNet(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        net = TestNet()
        input_tensor = tensor(np.random.rand(1, 3), requires_grad=True)
        output_tensor = net(input_tensor)
        loss = output_tensor.mean()
        loss.backward()

        for param in net.parameters():
            self.assertIsNotNone(param.grad)

        net.zero_grad()
        for param in net.parameters():
            self.assertIsNone(param.grad)

    def test_linear_forward_backward(self):
        # Compare with torch.nn.Linear
        in_features = 5
        out_features = 3
        batch_size = 2

        # pyml Linear
        linear_pyml = Linear(in_features, out_features)
        input_pyml = tensor(np.random.rand(batch_size, in_features), requires_grad=True)
        output_pyml = linear_pyml(input_pyml)
        grad_output_pyml = tensor(np.random.rand(batch_size, out_features))
        output_pyml.backward(grad_output_pyml)

        # PyTorch Linear
        linear_torch = nn.Linear(in_features, out_features)
        linear_torch.weight.data = torch.tensor(linear_pyml.weight.numpy())
        if linear_pyml.bias is not None:
            linear_torch.bias.data = torch.tensor(linear_pyml.bias.numpy())
        else:
            linear_torch.bias = None
        input_torch = torch.tensor(input_pyml.numpy(), requires_grad=True, dtype=torch.float32)
        output_torch = linear_torch(input_torch)
        grad_output_torch = torch.tensor(grad_output_pyml.numpy())
        output_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(output_pyml.numpy(), output_torch.detach().numpy(), rtol=1e-5, atol=1e-8))
        self.assertTrue(np.allclose(input_pyml.grad.numpy(), input_torch.grad.numpy(), rtol=1e-5, atol=1e-8))
        self.assertTrue(np.allclose(linear_pyml.weight.grad.numpy(), linear_torch.weight.grad.numpy(), rtol=1e-5, atol=1e-8))
        if linear_pyml.bias is not None and linear_torch.bias is not None:
            self.assertTrue(np.allclose(linear_pyml.bias.grad.numpy(), linear_torch.bias.grad.numpy(), rtol=1e-5, atol=1e-8))
        elif linear_pyml.bias is not None or linear_torch.bias is not None:
            self.fail("Bias presence mismatch between pyml and PyTorch Linear")

    def test_relu_forward_backward_in_module(self):
        # Compare with torch.nn.ReLU within a Module
        class TestReLUModule_pyml(Module):
            def __init__(self):
                super().__init__()
                self.relu = Relu()

            def forward(self, x):
                return self.relu(x)

        class TestReLUModule_torch(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        input_np = np.array([-1.0, 0.5, 0.0, 3.0], dtype=np.float32)
        grad_output_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # pyml
        relu_module_pyml = TestReLUModule_pyml()
        input_pyml = tensor(input_np, requires_grad=True)
        output_pyml = relu_module_pyml(input_pyml)
        output_pyml.backward(tensor(grad_output_np))

        # PyTorch
        relu_module_torch = TestReLUModule_torch()
        input_torch = torch.tensor(input_np, requires_grad=True)
        output_torch = relu_module_torch(input_torch)
        output_torch.backward(torch.tensor(grad_output_np))

        self.assertTrue(np.allclose(output_pyml.numpy(), output_torch.detach().numpy()))
        self.assertTrue(np.allclose(input_pyml.grad.numpy(), input_torch.grad.numpy()))

    def test_softmax_forward_backward_in_module(self):
        # Compare with torch.nn.Softmax within a Module
        class TestSoftmaxModule_pyml(Module):
            def __init__(self, dim):
                super().__init__()
                self.softmax = Softmax(dim=dim)
                self.dim = dim

            def forward(self, x):
                return self.softmax(x)

        class TestSoftmaxModule_torch(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.softmax = nn.Softmax(dim=dim)
                self.dim = dim

            def forward(self, x):
                return self.softmax(x)

        input_np = np.random.rand(2, 3).astype(np.float32)
        grad_output_np = np.random.rand(2, 3).astype(np.float32)
        dim = 1

        # pyml
        softmax_module_pyml = TestSoftmaxModule_pyml(dim)
        input_pyml = tensor(input_np, requires_grad=True)
        output_pyml = softmax_module_pyml(input_pyml)
        output_pyml.backward(tensor(grad_output_np))

        # PyTorch
        softmax_module_torch = TestSoftmaxModule_torch(dim)
        input_torch = torch.tensor(input_np, requires_grad=True)
        output_torch = softmax_module_torch(input_torch)
        output_torch.backward(torch.tensor(grad_output_np))

        self.assertTrue(np.allclose(output_pyml.numpy(), output_torch.detach().numpy(), rtol=1e-5, atol=1e-8))
        self.assertTrue(np.allclose(input_pyml.grad.numpy(), input_torch.grad.numpy(), rtol=1e-5, atol=1e-8))

if __name__ == '__main__':
    unittest.main()