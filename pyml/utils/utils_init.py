import numpy as np

def kaiming_uniform(fan_in, fan_out, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming uniform initialization (He initialization)
    
    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        a: Negative slope of the rectifier (default: 0 for LeakyReLU)
        mode: Either 'fan_in' (default) or 'fan_out'
        nonlinearity: Either 'relu' or 'leaky_relu' (default)
    """
    gain = np.sqrt(2.0) if nonlinearity == 'relu' else np.sqrt(2.0 / (1 + a ** 2))
    if mode == 'fan_in':
        bound = gain * np.sqrt(6.0 / fan_in)
    elif mode == 'fan_out':
        bound = gain * np.sqrt(6.0 / fan_out)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return np.random.uniform(-bound, bound, size=(fan_out, fan_in))

def xavier_uniform(fan_in, fan_out, gain=1.0):
    """
    Xavier/Glorot uniform initialization
    
    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        gain: Scaling factor
    """
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, size=(fan_out, fan_in))