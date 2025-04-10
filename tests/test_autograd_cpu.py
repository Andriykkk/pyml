import unittest
import numpy as np
from pyml.tensor import tensor
from pyml.device import Device
import torch

class TestAutogradAgainstPyTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        
        def zero_grad(t):
            if isinstance(t, (list, tuple)):
                for x in t:
                    zero_grad(x)
            else:
                t.grad = None
        
        self.zero_grad = zero_grad

    
    def compare_with_torch(self, pyml_func, torch_func, *inputs, rtol=1e-5, atol=1e-8):
        """
        Compare PyML and PyTorch implementations including forward and backward passes.
        
        Args:
            pyml_func: Function using PyML tensors
            torch_func: Equivalent function using PyTorch tensors
            inputs: Tuple of input values (will be converted to both PyML and PyTorch tensors)
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        pyml_inputs = []
        torch_inputs = []
        requires_grad = any(isinstance(x, (tuple, list)) and x[1] for x in inputs)
        
        for x in inputs:
            if isinstance(x, (tuple, list)):
                val, rg = x
                pyml_t = tensor(np.array(val), requires_grad=rg)
                torch_t = torch.tensor(val, dtype=torch.float32, requires_grad=rg)
            else:
                pyml_t = tensor(np.array(x))
                torch_t = torch.tensor(x, dtype=torch.float32)
            
            pyml_inputs.append(pyml_t)
            torch_inputs.append(torch_t)
        
        pyml_output = pyml_func(*pyml_inputs)
        torch_output = torch_func(*torch_inputs)
        
        self.assertTrue(
            np.allclose(pyml_output.numpy(), torch_output.detach().numpy(), rtol=rtol, atol=atol),
            f"Forward pass mismatch:\nPyML: {pyml_output.numpy()}\nPyTorch: {torch_output.detach().numpy()}"
        )
        
        if requires_grad:
            if pyml_output.shape == ():
                pyml_output.backward()
            else:
                pyml_output.backward(tensor(np.ones_like(pyml_output.numpy())))
            
            if torch_output.shape == ():
                torch_output.backward()
            else:
                torch_output.backward(torch.ones_like(torch_output))
            
            for pyml_in, torch_in in zip(pyml_inputs, torch_inputs):
                if pyml_in.requires_grad:
                    self.assertTrue(
                        np.allclose(pyml_in.grad.numpy(), torch_in.grad.numpy(), rtol=rtol, atol=atol),
                        f"Gradient mismatch for input:\nPyML: {pyml_in.grad.numpy()}\nPyTorch: {torch_in.grad.numpy()}"
                    )
            
            self.zero_grad(pyml_inputs)
            for t in torch_inputs:
                if t.grad is not None:
                    t.grad.zero_()
    
    def test_basic_arithmetic(self):
        # Addition
        self.compare_with_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            ([1.0, 2.0, 3.0], True),
            ([4.0, 5.0, 6.0], True)
        )
        
        # Subtraction
        self.compare_with_torch(
            lambda a, b: a - b,
            lambda a, b: a - b,
            ([1.0, 2.0, 3.0], True),
            ([4.0, 5.0, 6.0], True)
        )
        
        # Multiplication
        self.compare_with_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            ([1.0, 2.0, 3.0], True),
            ([4.0, 5.0, 6.0], True)
        )
        
        # Division
        self.compare_with_torch(
            lambda a, b: a / b,
            lambda a, b: a / b,
            ([1.0, 4.0, 9.0], True),
            ([1.0, 2.0, 3.0], True)
        )
    
    def test_broadcasting(self):
        # Broadcasting in addition
        self.compare_with_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            ([[1.0, 2.0], [3.0, 4.0]], True),
            ([1.0, 2.0], True)
        )
        
        # Broadcasting in multiplication
        self.compare_with_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            ([[1.0, 2.0], [3.0, 4.0]], True),
            ([1.0, 2.0], True)
        )
        
        # Broadcasting with scalar
        self.compare_with_torch(
            lambda a: a * 2.5,
            lambda a: a * 2.5,
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
    
    def test_matrix_operations(self):
        # Matrix multiplication
        self.compare_with_torch(
            lambda a, b: a @ b,
            lambda a, b: a @ b,
            ([[1.0, 2.0], [3.0, 4.0]], True),
            ([[5.0, 6.0], [7.0, 8.0]], True)
        )
        
        # Transpose
        self.compare_with_torch(
            lambda a: a.transpose(),
            lambda a: a.t(),
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
        
        # Batched matrix multiplication
        self.compare_with_torch(
            lambda a, b: a @ b,
            lambda a, b: a @ b,
            (np.random.rand(3, 4, 5), True),
            (np.random.rand(3, 5, 2), True)
        )
    
    def test_reduction_operations(self):
        # Sum all elements
        self.compare_with_torch(
            lambda a: a.sum(),
            lambda a: a.sum(),
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
        
        # Sum along axis
        self.compare_with_torch(
            lambda a: a.sum(axis=0),
            lambda a: a.sum(dim=0),
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
        
        # Sum with keepdims
        self.compare_with_torch(
            lambda a: a.sum(axis=1, keepdims=True),
            lambda a: a.sum(dim=1, keepdim=True),
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
        
        # Mean all elements
        self.compare_with_torch(
            lambda a: a.mean(),
            lambda a: a.mean(),
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
        
        # Mean along axis
        self.compare_with_torch(
            lambda a: a.mean(axis=0),
            lambda a: a.mean(dim=0),
            ([[1.0, 2.0], [3.0, 4.0]], True)
        )
    
    def test_complex_computations(self):
        # Test a more complex computation graph
        def pyml_computation(a, b, c):
            x = a * b
            y = x.sum() + c
            z = y * (a.mean() + b.max())
            return z
        
        def torch_computation(a, b, c):
            x = a * b
            y = x.sum() + c
            z = y * (a.mean() + b.max())
            return z
        
        self.compare_with_torch(
            pyml_computation,
            torch_computation,
            (np.random.rand(2, 3), True),
            (np.random.rand(2, 3), True),
            (5.0, True)
        )
        
        # Test with broadcasting and multiple operations
        def pyml_mixed_ops(a, b, c):
            x = (a + b) * c
            y = x.transpose() @ x
            return y.mean()
        
        def torch_mixed_ops(a, b, c):
            x = (a + b) * c
            y = x.t() @ x
            return y.mean()
        
        self.compare_with_torch(
            pyml_mixed_ops,
            torch_mixed_ops,
            (np.random.rand(3, 2), True),
            (np.random.rand(2), True),
            (np.random.rand(3, 2), True)
        )
    
    # def test_advanced_indexing(self):
    #     # Test with advanced indexing operations
    #     def pyml_indexing(a, b):
    #         x = a[[0, 1], :2] * b[1:, :2]
    #         return x.sum(axis=1).mean()
        
    #     def torch_indexing(a, b):
    #         x = a[[0, 1], :2] * b[1:, :2]
    #         return x.sum(dim=1).mean()
        
    #     self.compare_with_torch(
    #         pyml_indexing,
    #         torch_indexing,
    #         (np.random.rand(3, 3), True),
    #         (np.random.rand(3, 3), True)
    #     )
    
    def test_scalar_operations(self):
        # Test operations with scalar tensors
        self.compare_with_torch(
            lambda a, b: a * b + 3.0,
            lambda a, b: a * b + 3.0,
            (2.0, True),
            (3.0, True)
        )
        
        # Test reduction to scalar
        self.compare_with_torch(
            lambda a: (a * a).sum(),
            lambda a: (a * a).sum(),
            (np.random.rand(5), True)
        )
    
    def test_gradient_accumulation(self):
        # Test multiple backward passes accumulate gradients
        a_pyml = tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_pyml = tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        # First backward pass
        (a_pyml * b_pyml).sum().backward()
        (a_torch * b_torch).sum().backward()
        
        # Second backward pass (should accumulate)
        (a_pyml * b_pyml).sum().backward()
        (a_torch * b_torch).sum().backward()
        
        self.assertTrue(np.allclose(a_pyml.grad.numpy(), a_torch.grad.numpy()))
        self.assertTrue(np.allclose(b_pyml.grad.numpy(), b_torch.grad.numpy()))
    
    def test_no_grad(self):
        # Test operations with tensors that don't require grad
        self.compare_with_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            ([1.0, 2.0, 3.0], False),
            ([4.0, 5.0, 6.0], True)
        )