from pyml.tensor import tensor
import random
import numpy as np
import torch

class Dataset:
    def __init__(self, data, targets):
        """
        Custom Dataset class.

        Args:
            data: Tensor or array-like containing the data samples.
            targets: Tensor or array-like containing the corresponding targets.
        """
        if not isinstance(data, tensor):
            self.data = tensor(data)
        else:
            self.data = data

        if not isinstance(targets, tensor):
            self.targets = tensor(targets)
        else:
            self.targets = targets

        assert len(self.data) == len(self.targets), "Data and targets must have the same length."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Custom DataLoader class.

        Args:
            dataset: An iterable of your custom tensors OR a Dataset object
                     (with __len__ and __getitem__).
            batch_size: Number of samples to load in each batch.
            shuffle: If True, shuffle the data before creating batches.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset))) if hasattr(dataset, '__len__') else list(range(len(list(dataset))))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        if hasattr(self.dataset, '__getitem__'):
            for i in range(0, len(self.dataset), self.batch_size):
                batch_indices = self.indices[i : i + self.batch_size]
                batch = [self.dataset[idx] for idx in batch_indices]
                batch_data = [tensor(item[0].numpy()) if isinstance(item[0], torch.Tensor) else (item[0] if isinstance(item[0], tensor) else tensor(item[0])) for item in batch]
                batch_targets = [tensor(item[1]) if not isinstance(item[1], tensor) else item[1] for item in batch]
                stacked_data = self._stack_tensors(batch_data)
                stacked_targets = self._stack_tensors(batch_targets)
                yield stacked_data, stacked_targets
        else:
            batched_data = []
            batched_targets = []
            for idx in self.indices:
                item = list(self.dataset)[idx] 
                data, target = item[0], item[1]
                batched_data.append(data)
                batched_targets.append(target)
                if len(batched_data) == self.batch_size:
                    yield self._stack_tensors(batched_data), self._stack_tensors(batched_targets)
                    batched_data = []
                    batched_targets = []
            if batched_data:  
                yield self._stack_tensors(batched_data), self._stack_tensors(batched_targets)

    def __len__(self):
        if hasattr(self.dataset, '__len__'):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        else:
            return (len(list(self.dataset)) + self.batch_size - 1) // self.batch_size

    def _stack_tensors(self, tensors):
        """
        Stacks a list of tensors into a single tensor.
        Handles cases where tensors might have different shapes (e.g., the last batch).
        """
        if not tensors:
            return tensor(np.array([])) 

        first_shape = tensors[0].shape
        if all(t.shape == first_shape for t in tensors):
            return tensor(np.stack([t._data for t in tensors]), device=tensors[0].device, requires_grad=tensors[0].requires_grad)
        else:
            return tensors
