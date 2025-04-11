import unittest
import numpy as np
import torch
import torch.nn.functional as F
from pyml import tensor
from pyml.nn import Softmax

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax_fn = Softmax()
        self.softmax_dim_1 = Softmax()

    def test_softmax_forward(self):
        # Test basic functionality (default dim)
        x_pyml = tensor([1.0, 2.0, 0.5])
        y_pyml = self.softmax_fn(x_pyml)
        x_torch = F.softmax(torch.tensor([1.0, 2.0, 0.5]), dim=0).numpy()
        self.assertTrue(np.allclose(y_pyml.numpy(), x_torch))

        # Test with specified dimension
        x_pyml_batch = tensor([[1.0, 2.0, 0.5], [-1.0, 0.0, 1.0]])
        y_pyml_dim1 = x_pyml_batch.softmax(dim=1)
        x_torch_batch = F.softmax(torch.tensor([[1.0, 2.0, 0.5], [-1.0, 0.0, 1.0]]), dim=1).numpy()
        self.assertTrue(np.allclose(y_pyml_dim1.numpy(), x_torch_batch))

        y_pyml_dim0 = x_pyml_batch.softmax(dim=0)
        x_torch_batch_dim0 = F.softmax(torch.tensor([[1.0, 2.0, 0.5], [-1.0, 0.0, 1.0]]), dim=0).numpy()
        self.assertTrue(np.allclose(y_pyml_dim0.numpy(), x_torch_batch_dim0))

        # Test saved module (default dim)
        y2_pyml = self.softmax_fn(x_pyml)
        self.assertTrue(np.allclose(y2_pyml.numpy(), x_torch))

    def test_softmax_backward(self):
        # Test gradient computation (default dim)
        x_pyml = tensor([1.0, 2.0, 0.5], requires_grad=True)
        y_pyml = self.softmax_fn(x_pyml)
        grad_output_pyml = tensor([1.0, 0.5, 2.0])
        y_pyml.backward(grad_output_pyml)

        x_torch = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
        y_torch = F.softmax(x_torch, dim=0)
        grad_output_torch = torch.tensor([1.0, 0.5, 2.0])
        y_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(x_pyml.grad.numpy(), x_torch.grad.numpy()))

        # Test with specified dimension
        x_pyml_batch = tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
        y_pyml_dim1 = x_pyml_batch.softmax(dim=1)
        grad_output_pyml_batch = tensor([[1.0, 0.0], [0.5, 1.0]])
        y_pyml_dim1.backward(grad_output_pyml_batch)

        x_torch_batch = torch.tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
        y_torch_dim1 = F.softmax(x_torch_batch, dim=1)
        grad_output_torch_batch = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        y_torch_dim1.backward(grad_output_torch_batch)

        self.assertTrue(np.allclose(x_pyml_batch.grad.numpy(), x_torch_batch.grad.numpy()))

        x_pyml_batch.zero_grad()
        x_torch_batch.grad = None

        y_pyml_dim0 = x_pyml_batch.softmax(dim=0)
        grad_output_pyml_batch_dim0 = tensor([[0.2, 0.8], [1.1, 0.3]])
        y_pyml_dim0.backward(grad_output_pyml_batch_dim0)

        y_torch_dim0 = F.softmax(x_torch_batch, dim=0)
        grad_output_torch_batch_dim0 = torch.tensor([[0.2, 0.8], [1.1, 0.3]])
        y_torch_dim0.backward(grad_output_torch_batch_dim0)

        self.assertTrue(np.allclose(x_pyml_batch.grad.numpy(), x_torch_batch.grad.numpy()))

    def test_softmax_device_consistency(self):
        x_cpu = tensor([0.5, -0.5], device='cpu', requires_grad=True)
        y_cpu = x_cpu.softmax()
        y_cpu.backward(tensor([1.0, 1.0]))
        self.assertEqual(x_cpu.grad.device.type, 'cpu')

        x_cpu.zero_grad()
        y2_cpu = x_cpu.softmax(dim=0)
        y2_cpu.backward(tensor([2.0, 0.5]))
        self.assertEqual(x_cpu.grad.device.type, 'cpu')

    def test_softmax_memory(self):
        x = tensor(np.random.randn(100, 10), requires_grad=True)
        softmax_op = Softmax()
        y = softmax_op(x)
        y.backward(tensor(np.ones((100, 10))))
        del y
        self.assertTrue(x.grad is not None)

    def test_softmax_reference(self):
        x_pyml = tensor([0.1, -0.1], requires_grad=True)
        softmax_op = Softmax()

        y1_pyml = softmax_op(x_pyml)
        y1_pyml.backward(tensor([1.0, 1.0]))
        grad1_pyml = x_pyml.grad.numpy().copy()
        x_pyml.zero_grad()

        y2_pyml = softmax_op(x_pyml)
        y2_pyml.backward(tensor([1.0, 1.0]))
        grad2_pyml = x_pyml.grad.numpy()

        self.assertTrue(np.allclose(grad1_pyml, grad2_pyml))

    def test_softmax_functional(self):
        # Test the functional interface
        x_pyml = tensor([1.0, 2.0, 0.5], requires_grad=True)
        y_pyml = x_pyml.softmax()
        x_torch = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
        y_torch = F.softmax(x_torch, dim=0)
        self.assertTrue(np.allclose(y_pyml.numpy(), y_torch.detach().numpy()))

        grad_output_pyml = tensor([1.0, 0.5, 2.0])
        y_pyml.backward(grad_output_pyml)
        grad_output_torch = torch.tensor([1.0, 0.5, 2.0])
        y_torch.backward(grad_output_torch)
        self.assertTrue(np.allclose(x_pyml.grad.numpy(), x_torch.grad.numpy()))

        x_pyml_batch = tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
        y_pyml_dim1 = x_pyml_batch.softmax(dim=1)
        x_torch_batch = torch.tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
        y_torch_dim1 = F.softmax(x_torch_batch, dim=1)
        self.assertTrue(np.allclose(y_pyml_dim1.numpy(), y_torch_dim1.detach().numpy()))

        grad_output_pyml_batch = tensor([[1.0, 0.0], [0.5, 1.0]])
        y_pyml_dim1.backward(grad_output_pyml_batch)
        grad_output_torch_batch = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        x_torch_batch.grad = None # Reset gradients
        y_torch_dim1.backward(grad_output_torch_batch)
        self.assertTrue(np.allclose(x_pyml_batch.grad.numpy(), x_torch_batch.grad.numpy()))