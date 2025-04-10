import unittest
import torch
import numpy as np
from pyml.tensor import tensor
from pyml.utils import pytorch_to_pyml, pyml_to_pytorch, pytorch_dtype_to_pyml_dtype, pyml_dtype_to_pytorch_dtype

class TestTensorsUtils(unittest.TestCase):

    def setUp(self):
        """Setup function to run before each test."""
        self.pytorch_tensor = torch.randn(3, 3, requires_grad=True, device='cuda')
        
        self.pyml_tensor = tensor(data=self.pytorch_tensor.detach().cpu().numpy(),
                                dtype=pytorch_dtype_to_pyml_dtype(self.pytorch_tensor.dtype),
                                device=self.pytorch_tensor.device.type,
                                requires_grad=self.pytorch_tensor.requires_grad)

    def test_pytorch_to_pyml_conversion(self):
        """Test conversion from PyTorch tensor to custom tensor."""
        custom_tensor = pytorch_to_pyml(self.pytorch_tensor)

        np.testing.assert_array_equal(custom_tensor.numpy(), self.pytorch_tensor.detach().cpu().numpy())

        self.assertEqual(custom_tensor.dtype, pytorch_dtype_to_pyml_dtype(self.pytorch_tensor.dtype))

        self.assertEqual(custom_tensor.device, self.pytorch_tensor.device.type)

        self.assertEqual(custom_tensor.requires_grad, self.pytorch_tensor.requires_grad)

    def test_pyml_to_pytorch_conversion(self):
        """Test conversion from custom tensor to PyTorch tensor."""
        pytorch_tensor_converted = pyml_to_pytorch(self.pyml_tensor)

        np.testing.assert_array_equal(pytorch_tensor_converted.detach().cpu().numpy(), self.pyml_tensor.numpy())

        self.assertEqual(pytorch_tensor_converted.dtype, pyml_dtype_to_pytorch_dtype(self.pyml_tensor.dtype))

        self.assertEqual(pytorch_tensor_converted.device.type, self.pyml_tensor.device)

        self.assertEqual(pytorch_tensor_converted.requires_grad, self.pyml_tensor.requires_grad)

    def test_device_and_grad_preservation(self):
        """Test that the device and requires_grad properties are preserved across conversions."""
        # Test on a tensor created on CPU
        pytorch_tensor_cpu = torch.randn(3, 3, requires_grad=False, device='cpu')
        custom_tensor_cpu = pytorch_to_pyml(pytorch_tensor_cpu)
        pytorch_tensor_converted_cpu = pyml_to_pytorch(custom_tensor_cpu)

        self.assertEqual(custom_tensor_cpu.device, 'cpu')
        self.assertEqual(custom_tensor_cpu.requires_grad, False)
        self.assertEqual(pytorch_tensor_converted_cpu.device.type, 'cpu')
        self.assertEqual(pytorch_tensor_converted_cpu.requires_grad, False)

        # Test on a tensor created on CUDA
        pytorch_tensor_cuda = torch.randn(3, 3, requires_grad=True, device='cuda')
        custom_tensor_cuda = pytorch_to_pyml(pytorch_tensor_cuda)
        pytorch_tensor_converted_cuda = pyml_to_pytorch(custom_tensor_cuda)

        self.assertEqual(custom_tensor_cuda.device, 'cuda')
        self.assertEqual(custom_tensor_cuda.requires_grad, True)
        self.assertEqual(pytorch_tensor_converted_cuda.device.type, 'cuda')
        self.assertEqual(pytorch_tensor_converted_cuda.requires_grad, True)