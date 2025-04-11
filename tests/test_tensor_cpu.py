import unittest
import numpy as np
from pyml.tensor import tensor
from pyml.device import Device


class TestTensorCreation(unittest.TestCase):
    def test_tensor_from_list(self):
        data = [[1, 2, 3], [4, 5, 6]]
        t = tensor(data)
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.dtype, 'int64')   
        
    def test_tensor_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]])
        t = tensor(arr)
        self.assertEqual(t.shape, (2, 2))
        self.assertTrue(np.array_equal(t.numpy(), arr))
        
    def test_tensor_with_dtype(self):
        t = tensor([1, 2, 3], dtype='float32')
        self.assertEqual(t.dtype, 'float32')
        
    def test_tensor_device(self):
        t = tensor([1, 2, 3], device='cpu')
        self.assertEqual(t.device.type, 'cpu')


class TestFactoryMethods(unittest.TestCase):
    def test_zeros(self):
        t = tensor.zeros(2, 3)
        self.assertEqual(t.shape, (2, 3))
        self.assertTrue(np.all(t.numpy() == 0))
        
    def test_ones(self):
        t = tensor.ones(3, 2, dtype='float32')
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.dtype, 'float32')
        self.assertTrue(np.all(t.numpy() == 1))
        
    def test_rand(self):
        t = tensor.rand(5, 5)
        self.assertEqual(t.shape, (5, 5))
        self.assertTrue(np.all(t.numpy() >= 0))
        self.assertTrue(np.all(t.numpy() < 1))
        
    def test_randn(self):
        t = tensor.randn(1000)
        self.assertEqual(t.shape, (1000,))
        self.assertAlmostEqual(np.mean(t.numpy()), 0, delta=0.1)
        self.assertAlmostEqual(np.std(t.numpy()), 1, delta=0.1)
        
    def test_randint(self):
        t = tensor.randint(0, 10, (3, 3))
        self.assertEqual(t.shape, (3, 3))
        self.assertTrue(np.all(t.numpy() >= 0))
        self.assertTrue(np.all(t.numpy() < 10))
        
    def test_empty(self):
        t = tensor.empty(2, 2)
        self.assertEqual(t.shape, (2, 2))
        
    def test_full(self):
        t = tensor.full((3, 3), 5.5, dtype='float32')
        self.assertEqual(t.shape, (3, 3))
        self.assertEqual(t.dtype, 'float32')
        self.assertTrue(np.all(t.numpy() == 5.5))
        
    def test_arange(self):
        t = tensor.arange(0, 10, 2)
        self.assertTrue(np.array_equal(t.numpy(), np.array([0, 2, 4, 6, 8])))
        
    def test_eye(self):
        t = tensor.eye(3)
        expected = np.eye(3)
        self.assertTrue(np.array_equal(t.numpy(), expected))

class TestReshape(unittest.TestCase):
    def test_reshape_basic(self):
        t = tensor([1, 2, 3, 4, 5, 6])
        reshaped_t = t.reshape(2, 3)
        self.assertEqual(reshaped_t.shape, (2, 3))
        self.assertTrue(np.array_equal(reshaped_t.numpy(), np.array([[1, 2, 3], [4, 5, 6]])))

        t2 = tensor([[1, 2], [3, 4], [5, 6]])
        reshaped_t2 = t2.reshape(6)
        self.assertEqual(reshaped_t2.shape, (6,))
        self.assertTrue(np.array_equal(reshaped_t2.numpy(), np.array([1, 2, 3, 4, 5, 6])))

        t3 = tensor.rand(2, 3, 4)
        reshaped_t3 = t3.reshape(3, 8)
        self.assertEqual(reshaped_t3.shape, (3, 8))
        self.assertEqual(reshaped_t3.size, t3.size)

    def test_reshape_backward(self):
        t = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        reshaped_t = t.reshape(2, 2)
        grad_output = tensor([[1.0, 2.0], [3.0, 4.0]])
        reshaped_t.backward(grad_output)
        self.assertTrue(np.array_equal(t.grad.numpy(), np.array([1.0, 2.0, 3.0, 4.0])))

        t2 = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        reshaped_t2 = t2.reshape(4)
        grad_output_2 = tensor([1.0, 2.0, 3.0, 4.0])
        reshaped_t2.backward(grad_output_2)
        self.assertTrue(np.array_equal(t2.grad.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]])))

        t3 = tensor.rand(2, 3, requires_grad=True)
        reshaped_t3 = t3.reshape(3, 2)
        grad_output_3 = tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        reshaped_t3.backward(grad_output_3)
        self.assertEqual(t3.grad.shape, (2, 3))
        self.assertIsNotNone(t3.grad)

    def test_reshape_no_grad(self):
        t = tensor([1, 2, 3, 4])
        reshaped_t = t.reshape(2, 2)
        # Backward should not do anything as requires_grad is False
        reshaped_t.backward(tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertIsNone(t.grad)

    def test_reshape_invalid_shape(self):
        t = tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            t.reshape(2, 2)

        t2 = tensor([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            t2.reshape(5)


class TestProperties(unittest.TestCase):
    def test_shape_property(self):
        t = tensor.rand(2, 3, 4)
        self.assertEqual(t.shape, (2, 3, 4))
        
    def test_dtype_property(self):
        t = tensor([1, 2, 3], dtype='int32')
        self.assertEqual(t.dtype, 'int32')
        
    def test_ndim_property(self):
        t1 = tensor([1, 2, 3])
        self.assertEqual(t1.ndim, 1)
        
        t2 = tensor([[1, 2], [3, 4]])
        self.assertEqual(t2.ndim, 2)
        
    def test_size_property(self):
        t = tensor.rand(2, 3, 4)
        self.assertEqual(t.size, 24)

class TestDevice(unittest.TestCase):
    def test_device_class(self):
        dev = Device('cpu')
        self.assertEqual(dev.type, 'cpu')
        self.assertEqual(str(dev), 'cpu')
        
    def test_device_equality(self):
        dev1 = Device('cpu')
        dev2 = Device('cpu')
        self.assertEqual(dev1, dev2)
        self.assertEqual(dev1, 'cpu')
        
    def test_is_available(self):
        self.assertTrue(Device.is_available('cpu'))
        self.assertFalse(Device.is_available('cuda'))



class TestNumpyConversion(unittest.TestCase):
    def test_to_numpy(self):
        arr = np.random.rand(3, 3)
        t = tensor(arr)
        arr_back = t.numpy()
        self.assertTrue(np.array_equal(arr, arr_back))
        self.assertIsNot(arr, arr_back) 
        
    def test_array_interface(self):
        arr = np.random.rand(3, 3)
        t = tensor(arr)
        result = np.sum(t)
        expected = np.sum(arr)
        self.assertEqual(result, expected)

class TestRequiresGrad(unittest.TestCase):
    def test_requires_grad_default(self):
        t = tensor([1, 2, 3])
        self.assertFalse(t.requires_grad)
        
    def test_requires_grad_true(self):
        t = tensor([1, 2, 3], requires_grad=True)
        self.assertTrue(t.requires_grad)
        self.assertIsNone(t.grad)

class TestEdgeCases(unittest.TestCase):
    def test_empty_tensor(self):
        t = tensor([])
        self.assertEqual(t.shape, (0,))
        
    def test_scalar_tensor(self):
        t = tensor(5)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.numpy(), 5)
        
    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            tensor([1, 2, 3], device='invalid_device')
            
    def test_invalid_dtype(self):
        with self.assertRaises(TypeError):
            tensor([1, 2, 3], dtype='invalid_type')

class TestFactoryMethodEdgeCases(unittest.TestCase):
    def test_zeros_empty_shape(self):
        t = tensor.zeros(0)
        self.assertEqual(t.shape, (0,))
        
    def test_ones_scalar(self):
        t = tensor.ones(())
        self.assertEqual(t.shape, ())
        self.assertEqual(t.numpy(), 1)
        
    def test_randn_empty(self):
        t = tensor.randn(0)
        self.assertEqual(t.shape, (0,))
        
    def test_arange_no_stop(self):
        t = tensor.arange(5)
        self.assertTrue(np.array_equal(t.numpy(), np.array([0, 1, 2, 3, 4])))


# Operations
class TestTensorOperations(unittest.TestCase):
    def setUp(self):
        self.a = tensor([[1, 2], [3, 4]])
        self.b = tensor([[5, 6], [7, 8]])
        self.vec1 = tensor([1, 2, 3])
        self.vec2 = tensor([4, 5, 6])
        self.scalar = 2
        self.mat3d = tensor(np.random.rand(3, 4, 5))
        self.mat3d2 = tensor(np.random.rand(3, 5, 2))

    def test_addition(self):
        # Element-wise addition
        result = self.a + self.b
        expected = tensor([[6, 8], [10, 12]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Addition with scalar
        result = self.a + self.scalar
        expected = tensor([[3, 4], [5, 6]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Broadcasting
        a = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape (2, 2, 2)
        b = tensor([[1, 2], [3, 4]])  # shape (2, 2)
        result = a + b
        expected = tensor([[[2, 4], [6, 8]], [[6, 8], [10, 12]]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_subtraction(self):
        # Element-wise subtraction
        result = self.a - self.b
        expected = tensor([[-4, -4], [-4, -4]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Subtraction with scalar
        result = self.a - self.scalar
        expected = tensor([[-1, 0], [1, 2]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Broadcasting
        a = tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])  # shape (2, 2, 2)
        b = tensor([[1, 2], [3, 4]])  # shape (2, 2)
        result = a - b
        expected = tensor([[[9, 18], [27, 36]], [[49, 58], [67, 76]]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_multiplication(self):
        # Element-wise multiplication
        result = self.a * self.b
        expected = tensor([[5, 12], [21, 32]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Multiplication with scalar
        result = self.a * self.scalar
        expected = tensor([[2, 4], [6, 8]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Broadcasting
        a = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape (2, 2, 2)
        b = tensor([[1, 2], [3, 4]])  # shape (2, 2)
        result = a * b
        expected = tensor([[[1, 4], [9, 16]], [[5, 12], [21, 32]]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_division(self):
        # Element-wise division
        a = tensor([[1, 4], [9, 16]], dtype='float32')
        b = tensor([[1, 2], [3, 4]], dtype='float32')
        result = a / b
        expected = tensor([[1, 2], [3, 4]], dtype='float32')
        self.assertTrue(np.allclose(result.numpy(), expected.numpy()))
        
        # Division with scalar
        result = a / 2
        expected = tensor([[0.5, 2], [4.5, 8]], dtype='float32')
        self.assertTrue(np.allclose(result.numpy(), expected.numpy()))
        
        # Broadcasting
        a = tensor([[[1, 4], [9, 16]], [[25, 36], [49, 64]]], dtype='float32')  # shape (2, 2, 2)
        b = tensor([[1, 2], [3, 4]], dtype='float32')  # shape (2, 2)
        result = a / b
        expected = tensor([[[1, 2], [3, 4]], [[25, 18], [49/3, 16]]], dtype='float32')
        self.assertTrue(np.allclose(result.numpy(), expected.numpy()))

    def test_transpose(self):
        # 2D transpose
        a = tensor([[1, 2, 3], [4, 5, 6]])
        result = a.transpose()
        expected = tensor([[1, 4], [2, 5], [3, 6]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # 3D transpose
        a = tensor(np.arange(24).reshape(2, 3, 4))
        result = a.transpose(1, 0, 2)  # Swap first two dimensions
        expected = np.transpose(a.numpy(), (1, 0, 2))
        self.assertTrue(np.array_equal(result.numpy(), expected))
        
        # 4D transpose with custom axes
        a = tensor(np.arange(120).reshape(2, 3, 4, 5))
        result = a.transpose(2, 0, 3, 1)
        expected = np.transpose(a.numpy(), (2, 0, 3, 1))
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_matmul(self):
        # Vector dot product
        result = self.vec1.matmul(self.vec2)
        expected = tensor(32)  # 1*4 + 2*5 + 3*6 = 32
        self.assertEqual(result.numpy(), expected.numpy())
        
        # Matrix multiplication
        a = tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        b = tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
        result = a.matmul(b)
        expected = tensor([[9, 12, 15], [19, 26, 33], [29, 40, 51]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
        
        # Batched matrix multiplication
        a = tensor(np.random.rand(3, 4, 5))
        b = tensor(np.random.rand(3, 5, 2))
        result = a.matmul(b)
        expected = np.matmul(a.numpy(), b.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected, rtol=1e-5))
        
        # Operator @
        result = a @ b
        self.assertTrue(np.allclose(result.numpy(), expected, rtol=1e-5))

    def test_invalid_operations(self):
        # Shape mismatch for element-wise operations
        with self.assertRaises(ValueError):
            self.a + tensor([1, 2, 3])
        
        # Invalid matmul shapes
        with self.assertRaises(ValueError):
            self.a.matmul(tensor([1, 2, 3]))
        
        # Invalid types
        with self.assertRaises(TypeError):
            self.a + "invalid"
        
        with self.assertRaises(TypeError):
            self.a.matmul("invalid")

    def test_operation_with_different_dtypes(self):
        # Operations between different dtypes should promote to higher precision
        a = tensor([1, 2, 3], dtype='float32')
        b = tensor([1, 2, 3], dtype='float64')
        result = a + b
        self.assertEqual(result.dtype, 'float32')
        
        a = tensor([1, 2, 3], dtype='int32')
        b = tensor([1, 2, 3], dtype='float32')
        result = a * b
        self.assertEqual(result.dtype, 'int32')

    def test_negation(self):
        a = tensor([1, 2, 3])
        result = -a
        expected = tensor([-1, -2, -3])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_inplace_operations(self):
        # Note: These would require implementing __iadd__, etc. in the tensor class
        a = tensor([1, 2, 3])
        b = tensor([4, 5, 6])
        a += b
        self.assertTrue(np.array_equal(a.numpy(), [5, 7, 9]))
        
        a = tensor([1, 2, 3])
        a *= 2
        self.assertTrue(np.array_equal(a.numpy(), [2, 4, 6]))

# Indexing
class TestTensorOperations(unittest.TestCase):
    
    def test_single_index_access(self):
        t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t[0]   
        expected = tensor([1, 2, 3])  
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
    
    def test_slice_tensor(self):
        t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t[1:]  
        expected = tensor([[4, 5, 6], [7, 8, 9]])  
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
    
    def test_multi_dimensional_indexing(self):
        t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t[1][2]  
        expected = tensor(6)  
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))
    
    def test_chaining_indexing_and_slicing(self):
        t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t[1:][1]  
        expected = tensor([7, 8, 9]) 
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))