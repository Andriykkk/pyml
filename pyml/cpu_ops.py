import numpy as np

class CPUOps:
    @staticmethod
    def neg(val):
        from pyml.tensor import tensor
        return tensor(-val._data, dtype=val.dtype, device=val.device.type)

    @staticmethod
    def add(val1, val2):
        from pyml.tensor import tensor
        result_data = val1._data + val2._data
            
        result = tensor(result_data, dtype=val1.dtype, device=val1.device.type,
                    requires_grad=val1.requires_grad or val2.requires_grad)

        return result

    @staticmethod
    def sub(val1, val2):
        from pyml.tensor import tensor
        result_data = val1._data - val2._data
        return tensor(result_data, dtype=val1.dtype, device=val1.device.type,
                    requires_grad=val1.requires_grad or val2.requires_grad)

    @staticmethod
    def mul(val1, val2):
        from pyml.tensor import tensor
        result_data = val1._data * val2._data
        return tensor(result_data, dtype=val1.dtype, device=val1.device.type,
                    requires_grad=val1.requires_grad or val2.requires_grad)

    @staticmethod
    def div(val1, val2):
        from pyml.tensor import tensor
        result_data = val1._data / val2._data
        return tensor(result_data, dtype=val1.dtype, device=val1.device.type,
                    requires_grad=val1.requires_grad or val2.requires_grad)

    @staticmethod
    def matmul(val1, val2):
        from pyml.tensor import tensor
        result_data = np.matmul(val1._data, val2._data)
        return tensor(result_data, dtype=val1.dtype, device=val1.device.type,
                    requires_grad=val1.requires_grad or val2.requires_grad)

    @staticmethod
    def transpose(val, *axes):
        from pyml.tensor import tensor
        transposed_data = np.transpose(val._data, axes) if axes else np.transpose(val._data)
        return tensor(transposed_data, dtype=val.dtype, device=val.device.type,
                    requires_grad=val.requires_grad)

    @staticmethod
    def sum(val, axis=None, keepdims=False, **kwargs):
        from pyml.tensor import tensor
        result_data = np.sum(val._data, axis=axis, keepdims=keepdims, **kwargs)
        return tensor(result_data, dtype=val.dtype, device=val.device.type,
                    requires_grad=val.requires_grad)

    @staticmethod
    def max(val, axis=None, keepdims=False):
        from pyml.tensor import tensor
        result_data = np.max(val._data, axis=axis, keepdims=keepdims)
        return tensor(result_data, dtype=val.dtype, device=val.device.type,
                    requires_grad=val.requires_grad)

    @staticmethod
    def mean(val, axis=None, keepdims=False):
        from pyml.tensor import tensor
        result_data = np.mean(val._data, axis=axis, keepdims=keepdims)
        return tensor(result_data, dtype=val.dtype, device=val.device.type,
                    requires_grad=val.requires_grad)
    