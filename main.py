import numpy as np
from pyml.tensor import tensor

tensor_zeros = tensor.zeros(3, 3, dtype="int8") 
print("Tensor filled with zeros:")
print(tensor_zeros)
print(tensor_zeros.dtype)

a = tensor([1, 2, 3], dtype='int32')
b = tensor([1, 2, 3], dtype='float32')
result = a * b
print(result.dtype, result, result[0], 'int32')

t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = t[1][2]  
expected = tensor([6])  
print(result, expected)
self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))