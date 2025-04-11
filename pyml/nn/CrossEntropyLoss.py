import numpy as np
from pyml.tensor import tensor

_ops_crossentropy = {
    'cpu': {
        'forward': lambda logits, targets, reduction: _crossentropy_forward_cpu(logits, targets, reduction),
        'backward': lambda ctx, grad_output: _crossentropy_backward_cpu(ctx, grad_output)
    }
}

class CrossEntropyLoss:
    """CrossEntropyLoss that can be called directly"""
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def __call__(self, logits, targets):
        return self.forward(logits, targets)
    
    def forward(self, logits, targets):
        """Forward pass of cross entropy loss"""
        if not isinstance(logits, tensor):
            logits = tensor(logits)
        if not isinstance(targets, tensor):
            targets = tensor(targets)
            
        result = _ops_crossentropy[logits.device.type]['forward'](logits, targets, self.reduction)
        
        if result.requires_grad:
            result._ctx = (logits, targets, self.reduction)
            result._grad_fn = _ops_crossentropy[logits.device.type]['backward']
            
        return result

def cross_entropy(logits, targets, reduction='mean'):
    """Functional interface for cross entropy loss"""
    return CrossEntropyLoss(reduction=reduction)(logits, targets)

def _crossentropy_forward_cpu(logits, targets, reduction='mean'):
    
    log_softmax = logits._data - np.max(logits._data, axis=-1, keepdims=True)
    log_softmax = log_softmax - np.log(np.sum(np.exp(log_softmax), axis=-1, keepdims=True))
    
    if targets._data.ndim == 1:
        batch_size = targets._data.shape[0]
        loss = -log_softmax[np.arange(batch_size), targets._data]
    else:
        loss = -np.sum(targets._data * log_softmax, axis=-1)
    
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    
    return tensor(loss, dtype=logits.dtype,
                 device=logits.device.type,
                 requires_grad=logits.requires_grad)

def _crossentropy_backward_cpu(ctx, grad_output):
    logits, targets, reduction = ctx
    batch_size = logits._data.shape[0]
    
    shifted = logits._data - np.max(logits._data, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    
    if targets._data.ndim == 1:
        grad = probs.copy()
        grad[np.arange(batch_size), targets._data] -= 1
    else:
        grad = probs - targets._data
    
    if reduction == 'mean':
        grad /= batch_size
    
    logits.backward(tensor(grad, device=logits.device.type))

tensor.cross_entropy = lambda self, targets, reduction='mean': cross_entropy(self, targets, reduction)