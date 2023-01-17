

# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmmtl.core.utils import sync_random_seed
from mmdetection.mmdet.utils import get_device
from mmmtl.datasets import SAMPLERS

import math
from typing import TypeVar, Optional, Iterator, Iterable, Sequence, List, Generic, Sized, Union

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

class DistributedBatchSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 prefix=0,
                 batch_size=1,
                 recursive=False,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.batch_size=batch_size
        self.recursice = recursive
        device = get_device()
        self.seed = sync_random_seed(seed, device)
        self.is_batch_sampler = True

        self.prefix = prefix

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if self.recursice:
            raise NotImplementedError
        else:
            batch = []
            for idx in indices:
                batch.append(self.prefix + idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            yield batch

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

@SAMPLERS.register_module()
class Distributed_Weighted_BatchSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 batch_sizes,
                 weights,
                 num_samples = 32,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.datasets = dataset.datasets
        self.samplers = []
        self.sampleriters = []
        cur_len = 0
        for idx, dataset in enumerate(self.datasets):
            sampler = DistributedBatchSampler(dataset, batch_sizes[idx], prefix=cur_len,
                        num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
            self.samplers.append(sampler)
            self.sampleriters.append(iter(sampler))
            cur_len = cur_len + len(sampler)

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.batch_sizes = batch_sizes
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        self.shuffle = shuffle
        self.seed = seed

        device = get_device()
        self.seed = sync_random_seed(seed, device)
        self.is_batch_sampler = True
            

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        rand_tensor = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)

        for idx in rand_tensor:
            try:
                batch = next(self.sampleriters[idx])
            except:
                self.sampleriters[idx] = iter(self.samplers[idx])
                batch = next(self.sampleriters[idx])
            return batch


# import torch
# from torch.utils.data import DistributedSampler as _DistributedSampler

# from mmcls.core.utils import sync_random_seed
# from mmcls.datasets import SAMPLERS

# from .distributed_sampler import DistributedSampler

# import numpy as np
# # from . import dataset_dict
# from torch.utils.data import Dataset, Sampler
# from torch.utils.data import WeightedRandomSampler
# from typing import Optional
# from operator import itemgetter
# import torch

# class DatasetFromSampler(Dataset):
#     """Dataset to create indexes from `Sampler`.
#     Args:
#         sampler: PyTorch sampler
#     """

#     def __init__(self, sampler: Sampler):
#         """Initialisation for DatasetFromSampler."""
#         self.sampler = sampler
#         self.sampler_list = None

#     def __getitem__(self, index: int):
#         """Gets element of the dataset.
#         Args:
#             index: index of the element in the dataset
#         Returns:
#             Single element by index
#         """
#         if self.sampler_list is None:
#             self.sampler_list = list(self.sampler)
#         return self.sampler_list[index]

#     def __len__(self) -> int:
#         """
#         Returns:
#             int: length of the dataset
#         """
#         return len(self.sampler)


# @SAMPLERS.register_module()
# class DistributedSampler(_DistributedSampler):

#     def __init__(self,
#                  dataset,
#                  num_replicas=None,
#                  rank=None,
#                  shuffle=True,
#                  round_up=True,
#                  seed=0):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank)
#         self.shuffle = shuffle
#         self.round_up = round_up
#         if self.round_up:
#             self.total_size = self.num_samples * self.num_replicas
#         else:
#             self.total_size = len(self.dataset)

#         # In distributed sampling, different ranks should sample
#         # non-overlapped data in the dataset. Therefore, this function
#         # is used to make sure that each rank shuffles the data indices
#         # in the same order based on the same seed. Then different ranks
#         # could use different indices to select non-overlapped data from the
#         # same data list.
#         self.seed = sync_random_seed(seed)

#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         if self.shuffle:
#             g = torch.Generator()
#             # When :attr:`shuffle=True`, this ensures all replicas
#             # use a different random ordering for each epoch.
#             # Otherwise, the next iteration of this sampler will
#             # yield the same ordering.
#             g.manual_seed(self.epoch + self.seed)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()
#         else:
#             indices = torch.arange(len(self.dataset)).tolist()

#         # add extra samples to make it evenly divisible
#         if self.round_up:
#             indices = (
#                 indices *
#                 int(self.total_size / len(indices) + 1))[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         if self.round_up:
#             assert len(indices) == self.num_samples

#         return iter(indices)


