from enum import IntEnum, unique
import logging
from typing import List
from detectron2.config import configurable

from torch.utils.data import Dataset, Subset
import torch
from detectron2.data import MetadataCatalog
import detectron2.utils.comm  as comm
from .init_sampling_methods import balanced_look_ahead


@unique
class LabelMode(IntEnum):
    """An enum of properties of a data in active learning process"""

    UNLABELLED = 0
    """When the data is not yet labelled by the oracle
    """

    LABELLED = 1
    """When the data is considered as labelled by the oracle
    """

    HIDDEN = -1
    """When the data is considered as remove in the dataset
    """


class ActiveDataset(Dataset):
    """Manages a dataset that will simulate the active learning training
    Args:
        cfg (CfgNode): The all-in-one detectron config file
        dataset (Dataset) : the original dataset
        
    Attributes:
        dataset (Dataset): the labelled dataset
        pool (Dataset): the unlabelled data that can be annoated
        init_indices (torch.Tensor): indices of the initial dataset
        start_n : number of data to label
        method : sampling method to be used for acquiring initial data
        starter: Actively rank the initial data on the pool
    """

    @configurable
    def __init__(
        self,
        dataset: Dataset,
        *,
        start_n: int,
        classes: List[str] = None,
        init_method: str = "Random",
        init_heuristic: str = None,
        starter = None,
        seed: int = None,
    ):

        self._original = dataset
        self._classlist = classes
        self.logger = logging.getLogger("activedet.data.manager")

        self.seed = seed
        if seed is None:
            self.seed = comm.shared_random_seed()

        # It's identical to reset, but written here for readibility
        self.labeller = torch.as_tensor([LabelMode.UNLABELLED] * len(self._original))
        self.init_indices = None
        self.starter = starter

        self.start_n = start_n
        self.method = init_method
        self.heuristic = init_heuristic

        self.start()

    @classmethod
    def from_config(cls, cfg, dataset):
        ret = {}
        ret["dataset"] = dataset
        ret["classes"] = MetadataCatalog.get(cfg["DATASETS"]["TRAIN"][0]).thing_classes
        ret["start_n"] = cfg.ACTIVE_LEARNING.START_N
        ret["init_method"] = cfg.DATASETS.INIT_SAMPLING_METHOD.NAME
        ret["init_heuristic"] = cfg.DATASETS.INIT_SAMPLING_METHOD.HEURISTIC
        ret["seed"] = cfg.SEED

        return ret

    def __getitem__(self, index):
        return self._original[self.labelled_idx[index]]

    def __len__(self):
        return len(self.labelled_idx)

    def start(self) -> None:
        """Start labelling the intial dataset
        Args:
            n : number of data to label
            method : sampling method to be used for acquiring initial data
        """
        g = torch.Generator()
        g.manual_seed(self.seed)
        if self.starter is not None:
            initials = self.starter.start(self.pool)
            # Release the model since it's only used once
            del self.starter
        elif self.method == "Random":
            initials = torch.randperm(len(self.pool_idx), generator=g)[:self.start_n]
        elif self.method == "BalancedLookAhead":
            initials = balanced_look_ahead(
                self.pool.unlabelled, self._classlist, self.start_n, self.heuristic, generator=g
            )
        else:
            raise NotImplementedError(
                f"No sampling method named as {self.method} is implemented."
            )
        self.init_indices = self.pool_idx[initials]
        self.acquire(initials)

    def remove_start(self):
        """Remove the starting indices in the current labelled dataset"""
        for i in self.init_indices:
            self.labeller[i] = LabelMode.HIDDEN

        self.logger.info(
            f"Successfully removed {len(self.init_indices)} initial samples. Current size is {len(self)}"
        )

    def acquire(self, indices: List[int]) -> None:
        """Acquire the labels based on the given index
        Args:
            indices = indices of pool data that will be acquired
        Notes:
            The indices here is based on the arrangement in the current pool Dataset
            It is not the absolute index with respect to the original dataset
        """
        valid_idx = self.pool_idx[indices]

        for i in valid_idx:
            self.labeller[i] = LabelMode.LABELLED

        self.logger.info(
            f"Successfully added {len(valid_idx)} initial samples. Total size is {len(self)}"
        )

    @property
    def pool(self):
        return Pool(self._original, self.pool_idx)

    @property
    def dataset(self):
        return Subset(self._original, self.labelled_idx)

    @property
    def labelled_idx(self):
        return (self.labeller == LabelMode.LABELLED).nonzero()

    @property
    def pool_idx(self):
        return (self.labeller == LabelMode.UNLABELLED).nonzero()

    def reset(self) -> None:
        """Resets the labels into its original state"""
        self.labeller = torch.as_tensor([LabelMode.UNLABELLED] * len(self._original))
        self.init_indices = None
        self.logger.info(
            f"Successful reset on dataset. Current size is {len(self)}"
        )

    def state_dict(self):
        """Returns a state_dict that can be saved in a checkpoint together with other state_dict"""
        return {
            "labeller": self.labeller,
        }

    def load_state_dict(self, state_dict):
        """Loads a state dict. Typically, it loads from a checkpoint"""
        self.labeller = state_dict["labeller"]


class Pool(Dataset):
    """It emulates a pool in Active Learning Process
    wherein each data point does not contain any labels
    """

    def __init__(self, original, pool_indices):
        self.unlabelled = Subset(original, pool_indices)

    def __len__(self):
        return len(self.unlabelled)

    def __getitem__(self, index):
        # Return the data only. No label
        return {
            key: value
            for key, value in self.unlabelled[index].items()
            if key != "annotations"
        }
