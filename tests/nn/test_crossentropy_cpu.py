import unittest
import numpy as np
import torch
import torch.nn as nn
from pyml import tensor
from pyml.nn import CrossEntropyLoss, cross_entropy
import torch.nn.functional as F

class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.criterion_pyml = CrossEntropyLoss()
        self.criterion_torch = nn.CrossEntropyLoss()
    
    def test_crossentropy_forward_class_indices(self):
        # Test with class indices
        logits = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype='float32')
        targets = tensor([0, 1], dtype='int64')
        
        loss_pyml = self.criterion_pyml(logits, targets)
        loss_torch = self.criterion_torch(torch.tensor(logits.numpy()), 
                                        torch.tensor(targets.numpy()))
        
        self.assertAlmostEqual(loss_pyml.numpy(), loss_torch.item(), places=5)
    
    def test_crossentropy_forward_one_hot(self):
        # Test with one-hot encoded targets
        logits = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype='float32')
        targets_pyml = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')

        loss_pyml = self.criterion_pyml(logits, targets_pyml)

        logits_torch = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        targets_torch = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        log_probs_torch = torch.log_softmax(logits_torch, dim=1)
        loss_torch = -(targets_torch * log_probs_torch).sum(dim=1).mean().item()

        self.assertAlmostEqual(loss_pyml.numpy(), loss_torch, places=5)
    
    def test_crossentropy_backward_class_indices(self):
        # Test gradient computation with class indices
        logits = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
        targets = tensor([0, 1])
        
        loss = self.criterion_pyml(logits, targets)
        loss.backward()
        
        logits_torch = torch.tensor(logits.numpy(), requires_grad=True)
        loss_torch = self.criterion_torch(logits_torch, 
                                        torch.tensor(targets.numpy()))
        loss_torch.backward()
        
        self.assertTrue(np.allclose(logits.grad.numpy(), 
                                  logits_torch.grad.numpy(), 
                                  rtol=1e-5))
    
    def test_crossentropy_backward_one_hot(self):
        # Test gradient computation with one-hot encoded targets
        logits_pyml = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True, dtype='float32')
        targets_pyml = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')

        loss_pyml = self.criterion_pyml(logits_pyml, targets_pyml)
        loss_pyml.backward()

        logits_torch = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
        targets_torch = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        log_probs_torch = F.log_softmax(logits_torch, dim=1)
        loss_torch = -(targets_torch * log_probs_torch).sum(dim=1).mean()
        loss_torch.backward()

        self.assertTrue(np.allclose(logits_pyml.grad.numpy(),
                                  logits_torch.grad.numpy(),
                                  rtol=1e-5))
        
    def test_crossentropy_reduction(self):
        # Test different reduction methods
        logits = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        targets = tensor([0, 1])
        
        # Test mean reduction (default)
        loss_mean = CrossEntropyLoss(reduction='mean')(logits, targets)
        loss_mean_torch = nn.CrossEntropyLoss(reduction='mean')(
            torch.tensor(logits.numpy()), 
            torch.tensor(targets.numpy()))
        self.assertAlmostEqual(loss_mean.numpy(), loss_mean_torch.item(), places=5)
        
        # Test sum reduction
        loss_sum = CrossEntropyLoss(reduction='sum')(logits, targets)
        loss_sum_torch = nn.CrossEntropyLoss(reduction='sum')(
            torch.tensor(logits.numpy()), 
            torch.tensor(targets.numpy()))
        self.assertAlmostEqual(loss_sum.numpy(), loss_sum_torch.item(), places=5)
    
    def test_crossentropy_numerical_stability(self):
        # Test with large logits
        logits = tensor([[1000.0, 1001.0, 1002.0]])
        targets = tensor([0])
        
        loss = self.criterion_pyml(logits, targets)
        self.assertTrue(np.isfinite(loss.numpy()))
        
        # Test with very small logits
        logits = tensor([[-1000.0, -1001.0, -1002.0]])
        targets = tensor([0])
        
        loss = self.criterion_pyml(logits, targets)
        self.assertTrue(np.isfinite(loss.numpy()))
    
    def test_crossentropy_functional_interface(self):
        # Test functional interface
        logits = tensor([[1.0, 2.0, 3.0]])
        targets = tensor([0])
        
        loss_func = cross_entropy(logits, targets)
        loss_class = CrossEntropyLoss()(logits, targets)
        
        self.assertAlmostEqual(loss_func.numpy(), loss_class.numpy(), places=5)
    
    def test_crossentropy_tensor_method(self):
        # Test tensor method interface
        logits = tensor([[1.0, 2.0, 3.0]])
        targets = tensor([0])
        
        loss_method = logits.cross_entropy(targets)
        loss_class = CrossEntropyLoss()(logits, targets)
        
        self.assertAlmostEqual(loss_method.numpy(), loss_class.numpy(), places=5)