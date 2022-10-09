from typing import Iterator
import math
import itertools

import torch
from torch.utils.data.sampler import Sampler, T_co

from detectron2.utils import comm

from activedet.utils import partition, divide


class InferenceRepeatSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Additionally, it repeats the sampling of dataset that is useful for Monte Carlo Estimation

    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.

    What happens when it is unevenly distributed?
    For example num_repeats = 10, It will definitely create uneven distribution

    For this implementation, we add padding at the end similar to DistributedSampler
    Source:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler


    Attributes:
        num_replicas (int) : Number of GPUs devices accomodated by this Sampler
        num_size (int) : original dataset size
        total_size (int): Minimum dataset size that is compatible for dist traininig.
                It is calculated as the total dataset size (inclusive of the repeats).

    Known Issue:
        InferenceRepeatSampler doesn't work well when num_repeat is smaller than half of world size.
        The reason is because of interlacing issue. Consider the ff scenario:
            dataset = [A, B, C, D, E, F, G, H, I, J]
            num_repeats = 1
            num_gpus (world size) = 4
        Then, it will be interlaced on the GPUs as follows:
            GPU     DATA STREAM
            0       A,  E,  I
            1       B,  F,  J
            2       C,  G,  I (padding)
            3       D,  H,  J (padding)
            (Note: It is assummed that the user will remove the padding altogether)
        Having seen this, this is how each GPU will process each images.

        Consider now that we call the ff:
            ```python
                all_pred = comm.all_gather(pred)
                all_pred_flatten = list(itertools.chain.from_iterable(all_pred))
            ```
            where: pred is the variable that contains the prediction per GPU.

        Therefore, on the flatten prediction (from all GPUs), we expect the ff order:
            all_pred_flatten = [A, E, I, B, F, J, C, G, D, H] 
            (Note: padding is expected to be removed now)
        
        The main reason is because of interlacing performed in this sampling for optimization sake.


    """

    def __init__(self, size: int, num_repeats: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            num_repeats (int): Number of repeats to be performed on each data point
        """
        self._size = size
        assert size > 0
        assert num_repeats > 0
        assert (
            num_repeats > comm.get_world_size() // 2
        ), f"Ensure that {num_repeats} > {comm.get_world_size//2}. It's to avoid interlacing issue for preserving order."

        # Repeat images for monte carlo
        # e.g. num_size = 2
        #      num_repeats = 30
        #      order (dim 0) = [A B]
        # after repeat size = 60
        #       order (dim 0) = [A A A A .. A A B B B B B .. B B]
        self._repeated_indices = (
            torch.arange(size).repeat_interleave(num_repeats).tolist()
        )
        self._repeated_size = size * num_repeats

        self._rank = comm.get_rank()
        self.num_replicas = comm.get_world_size()
        self.num_size = size
        self.total_size = (
            math.ceil(self._repeated_size / self.num_replicas) * self.num_replicas
        )

    def __iter__(self) -> Iterator[T_co]:
        padding_size = self.total_size - self._repeated_size
        if padding_size <= len(self._repeated_indices):
            self._repeated_indices += self._repeated_indices[:padding_size]
        else:
            self._repeated_indices += (
                self._repeated_indices
                * math.ceil(padding_size / len(self._repeated_indices))
            )[:padding_size]

        assert (
            len(self._repeated_indices) == self.total_size
        ), f"The two indices must be equal length: {len(self._repeated_indices)} {self.total_size}"

        # De-interleave so that it will have the ff. COnsider that we have four gpus, using the example above:
        # GPU rank 0: A A A A A A A A B B B B B B B B
        # GPU rank 1: A A A A A A A A B B B B B B B B
        # GPU rank 2: A A A A A A A B B B B B B B B B
        # GPU rank 3: A A A A A A A B B B B B B B B B
        sharded_group = [
            self._repeated_indices[idx :: self.num_replicas]
            for idx in range(self.num_replicas)
        ]

        # It was already partitiioned according to the world size.
        local_indices = sharded_group[self._rank]

        yield from local_indices

    def __len__(self) -> int:
        return self.num_size

class TrivialSampler(Sampler):
    """It's similar to `InferenceSampler` but preserves order during concatenation
        Consider scenario #1:
            dataset = [A, B, C, D, E, F, G, H, I, J]
            num_gpus (world size) = 4
            batch_size = 16

            Then, the data distribution across GPUs are as follows:
                GPU     DATA BATCH
                0       A,  B,  C
                1       D,  E,  F
                2       G,  H
                3       I,  J
        
        Consider scenario #2:
            dataset = [A, B, C, D, E, F, G, H, I, J, K, L]
            num_gpus (world size) = 2
            batch_size = 8

            Then, the data distribution across GPUs are as follows:
                GPU     BATCH 1             BATCH 2
                0       A,  B,  C, D        I,  J
                1       E,  F,  G, H        K,  L
            Having seen this, this is how each GPU will process each batch.

        Example:
            In this example, we assume it's going to be run in two GPUs
            ```python
                from torch.utils.data import BatchSampler, DataLoader, TensorDataset
                import torch.distributed as dist

                dataset = TensorDataset(['A','B','C','D','E','F','G','H','I','J','K','L'])
                sampler = TrivialSampler(len(dataset))
                batch_sampler = BatchSampler(sampler,batch_size=8, drop_last=False)
                data_loader = DataLoader(dataset, num_workers=2, batch_sampler=batch_sampler)

                for batch in data_loader:
                    all_batch = dist.all_gather(batch)
                    batch_flatten = torch.cat(all_batch, dim=0)

                    print(batch_flatten)
                    # On first iteration, ['A','B','C','D','E','F','G','H']
                    # On second iteration, ['I','J','K','L']
            ```
    """
    def __init__(self, dataset_size: int, batch_size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            batch_size (int): total batch size across all workers. 
        """
        self._size = dataset_size
        self._batch_size = batch_size
        assert dataset_size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        pass

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(self._size))
        # [A, B, C, D, E, F, G, H, I, J, K, L] -> [[A, B, C, D, E, F, G, H], [I, J, K, L]]
        batch_chunks = partition(indices, self._batch_size)
        # Divide each chunk for each device
        # [[A, B, C, D, E, F, G, H], [I, J, K, L]] -> [ [[A,B,C,D],[E,F,G,H]], [[I,J], [K,L]] ]
        ranks_sliced = [divide(self._world_size,batch) for batch in batch_chunks]

        # Flatten into 1D
        # [ [[A,B,C,D],[E,F,G,H]], [[I,J], [K,L]] ] -> [ [A,B,C,D],[E,F,G,H], [I,J], [K,L] ]
        ranks_sliced = itertools.chain.from_iterable(ranks_sliced)

        # Order the slice per rank basis:
        # Original: [[A, B, C, D], [E, F, G, H], [I, J], [K, L]]
        # Rank 0: [[A, B, C, D], [I, J]]
        # Rank 1: [[E, F, G, H], [K, L]]
        ordered_batch = itertools.islice(ranks_sliced,self._rank,None,self._world_size)

        # Rank 0: [[A, B, C, D], [I, J]] -> [A, B, C, D, I, J]
        # Rank 1: [[E, F, G, H], [K, L]] -> [E, F, G, H, K, L]
        ordered_batch_flat = itertools.chain.from_iterable(ordered_batch)

        yield from ordered_batch_flat

    def __len__(self) -> int:
        return self._size // self._world_size
