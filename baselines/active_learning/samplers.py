
import numpy as np
from torch.utils.data.sampler import Sampler
import torch
from collections import Counter

class HalfSampler(Sampler):
    """
    At every iteration, this will first sample half of the batch, and then
    fill the other half of the batch with the same label distribution.
    batch_size must be an even number
    """
    def __init__(self, labels, batch_size, upsample = None):
        assert (batch_size % 2 == 0), "batch_size must be an even number"

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.batch_size = int(batch_size)
        self.index_to_label = labels
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        # upsampling the indices
        self.upsample = upsample
        if upsample is not None:
            # duplicate samples according to the counts
            self.all_indices = np.repeat(np.arange(len(labels)), upsample)
        else:
            self.all_indices = np.arange(len(labels))
        # sample half of the batch_size as self.length_of_single_pass
        self.length_of_single_pass = self.batch_size // 2
        self.batch_num = len(self.all_indices) // self.length_of_single_pass + 1
        self.list_size = len(self.all_indices) * 2

        assert self.list_size >= self.batch_size
    
    def __len__(self):
        return self.list_size
    
    def __iter__(self):
        idx_list = [0] * self.list_size
        num_iters = self.calculate_num_iters()
        
        # one pass of training data with size n
        n = len(self.all_indices)
        # self.all_indices may have repeated items after upsampling
        indices = torch.randperm(n).numpy()
        perm_indices = self.all_indices[indices]
        i = 0 # index the idx_list
        k = 0 # index the perm_indices
        for bcnt in range(num_iters):
            if bcnt < num_iters - 1:
                step =  self.length_of_single_pass
            else:
                step = len(self.index_to_label) % self.length_of_single_pass
            half_batch_indices = perm_indices[k: k + step]
            k += step
            idx_list[i : i + step] = half_batch_indices
            i += step
            # sample the other half with the same label distribution
            label_counts = Counter(self.index_to_label[half_batch_indices])
            for label, count in label_counts.items():
                t = self.labels_to_indices[label]
                idx_list[i : i + count] = c_f.safe_random_choice(
                    t, size=count
                )
                i += count
        return iter(idx_list)

    def calculate_num_iters(self):
        return self.batch_num

