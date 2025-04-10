from pyml.tensor import tensor
import torch

def pytorch_dtype_to_pyml_dtype(pytorch_dtype):
    """
    Converts a PyTorch dtype to your custom dtype.
    
    Args:
        pytorch_dtype: The PyTorch dtype to convert.
        
    Returns:
        str: The converted custom dtype.
    """
    dtype_map = {
        'torch.float32': 'float32',
        'torch.float64': 'float64',
        'torch.int32': 'int32',
        'torch.int64': 'int64',
        'torch.bool': 'bool',
        'torch.uint8': 'uint8',
        'torch.int8': 'int8',
        'torch.int16': 'int16',
    }
    
    return dtype_map[str(pytorch_dtype)]

def pyml_dtype_to_pytorch_dtype(pyml_dtype):
    """
    Converts a custom dtype to a PyTorch dtype.
    
    Args:
        pyml_dtype: The custom dtype to convert.
        
    Returns:
        torch.dtype: The converted PyTorch dtype.
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': torch.bool,
        'uint8': torch.uint8,
        'int8': torch.int8,
        'int16': torch.int16,
    }
    
    return dtype_map[pyml_dtype]

def pytorch_to_pyml(pytorch_tensor):
    """
    Converts a PyTorch tensor to your custom tensor class.
    
    Args:
        pytorch_tensor: The PyTorch tensor to convert.
        
    Returns:
        tensor: The converted custom tensor.
    """

    if not isinstance(pytorch_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    data = pytorch_tensor.detach().cpu().numpy()
    dtype = pytorch_dtype_to_pyml_dtype(pytorch_tensor.dtype)
    device = pytorch_tensor.device.type
    requires_grad = pytorch_tensor.requires_grad
    
    # Create and return the custom tensor
    return tensor(data=data, dtype=dtype, device=device, requires_grad=requires_grad)

def pyml_to_pytorch(pyml_tensor):
    """
    Converts a custom tensor to a PyTorch tensor.
    
    Args:
        pyml_tensor: The custom tensor to convert.
        
    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """

    if not isinstance(pyml_tensor, tensor):
        raise TypeError("Input must be a custom tensor")
    pytorch_tensor = torch.tensor(pyml_tensor._data, dtype=pyml_dtype_to_pytorch_dtype(pyml_tensor.dtype), device=pyml_tensor.device.type)
    
    if pyml_tensor.requires_grad:
        pytorch_tensor.requires_grad_(True)
    
    return pytorch_tensor