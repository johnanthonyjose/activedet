from typing import Callable, Dict, List
from functools import partial
import logging
import torch
from torch.utils.data import Subset
import detectron2.utils.comm as comm
from detectron2.config import configurable

from activedet.data.manager import ActiveDataset
from activedet.acquisition import build_heuristic
from activedet.pool import parse_features


class PoolRanker:
    """Responsible for letting a model ranks the pool according to its heuristic
    Attributes:
        pool_evaluator(Callable): responsible for the inference and post-processing of the pool
        heuristic (Heuristic):Defines how to measure the importance of a data point
        max_sample (int) : Amount of samples to observe on the pool.
            When it is -1, it takes all data available in the pool
    """

    def __init__(
        self,
        pool_evaluator: Callable,
        heuristic: Callable,
        *,
        ndata_to_label: int,
        max_sample: int,
        seed=None,
    ):

        self.heuristic = heuristic
        self.pool_evaluator = pool_evaluator
        self.ndata_to_label = ndata_to_label
        self.seed = seed
        if seed is None:
            self.seed = comm.shared_random_seed()
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.max_sample = max_sample
        if self.max_sample != -1:
            assert (
                self.ndata_to_label <= self.max_sample
            ), f"Nunber of labels to add {self.ndata_to_label} should be less than or equal to max sampling {self.max_sample}"

    def rank(self, pool) -> torch.Tensor:
        """Calculates which of the data points in the pool are important according to the heurstic
        Args:
            pool (Pool) : existing pool dataset where we are interested to rank
        Returns:
            indices of data from the pool that are ranked according to importance
        """
        logger = logging.getLogger("activedet.engine.ActiveDatasetUpdater")

        # Limit number of samples
        if self.max_sample != -1 and self.max_sample < len(pool):
            indices = torch.randperm(len(pool), generator=self.generator)[
                : self.max_sample
            ]
            pool = Subset(pool, indices)
            logger.info(f"subsampling the pool size into {len(indices)}")
        else:
            indices = torch.arange(len(pool))

        logger.info(f"Current pool size is {len(pool)}")
        instances = self.pool_evaluator(pool)
        logger.info(f"Done pool inference. Working on calculating the heursitics...")
        label_ranks = self.heuristic(instances)
        logger.info(f"Done on calculating heuristics.")

        to_label = indices[label_ranks]

        return to_label

class PoolRankStarter(PoolRanker):
    def __init__(
        self,
        pool_evaluator: Callable,
        heuristic: Callable,
        *,
        start_n: int,
        max_sample: int,
        seed=None,
        pool_transform=None,
    ):
        super().__init__(
            pool_evaluator=pool_evaluator,
            heuristic=heuristic,
            ndata_to_label=start_n,
            max_sample=max_sample,
            seed=seed,
        )
        self.pool_transform = pool_transform


    def start(self, pool) -> None:
        """Initializes the dataset according to the start heuristic
        It will drop the existing labeled data then re-initialize according to start_heuristic
        Args:
            dataset : the current dataset that needs to be updated.
                It is assumed that it is the same object across the whole training
        Notes:
            It assumes that this will only be ran once across the whole training
        """
        temp_transform = pool.unlabelled.dataset.transform

        if self.pool_transform:
            pool.unlabelled.dataset.transform = self.pool_transform

        to_label = self.rank(pool)
        init_indices = to_label[: self.ndata_to_label]

        pool.unlabelled.dataset.transform = temp_transform
        return init_indices



class ActiveDatasetUpdater(PoolRanker):
    """Adds the Active Learning step during training
    It's similar to the ActiveLearningLoop of Baal

    Attributes:
        pool_evaluator : responsible for the inference and post-processing of the pool
        heuristic : A function that defines how to measure the importance of a data point
        ndata_to_label : Amount of data to acquire for each acquisition step
        max_sample : Amount of samples to observe on the pool. When it is -1, it takes all data available in the pool
        train_out_features: If this is set, it acquires the intermediate features from the backbone.
            It will be passed on to the heuristic as an attribute. This list should contain the name of intermediate features
    """
    @configurable
    def __init__(
        self,
        pool_evaluator: Callable,
        heuristic: Callable,
        *,
        ndata_to_label: int,
        max_sample: int,
        seed=None,
        train_out_features: List[str] = [],
    ):
        super().__init__(
            pool_evaluator=pool_evaluator,
            heuristic=heuristic,
            ndata_to_label=ndata_to_label,
            max_sample=max_sample,
            seed=seed,
        )
        self.train_out_features = train_out_features

    @classmethod
    def from_config(cls, cfg, pool_estimator):
        return {
            "pool_evaluator": pool_estimator,
            "heuristic": build_heuristic(cfg),
            "ndata_to_label": cfg.ACTIVE_LEARNING.NDATA_TO_LABEL,
            "max_sample": cfg.ACTIVE_LEARNING.POOL.MAX_SAMPLE,
            "seed": cfg.SEED,
        }


    def extract_train_features(
        self, dataset: ActiveDataset, out_features
    ) -> Dict[str, torch.Tensor]:
        """Extracts each training point as a feature vector
        Args:
            dataset : active Dataset
        Returns:
            A dictionary of features wherein each key represents image id while the value represents the feature vector
        """
        temp_preprocessor = self.pool_evaluator.collator.preprocessor
        self.pool_evaluator.collator.preprocessor = partial(
            parse_features, nodes=out_features
        )
        centers = {}
        for features in self.pool_evaluator(dataset):
            centers.update(features)
        assert len(centers) == len(dataset)
        self.pool_evaluator.collator.preprocessor = temp_preprocessor
        return centers

    def update(self, dataset: ActiveDataset) -> None:
        """Updates the ranking for the pool after the training loop using the heuristics
        It is equivalent to the step method in ActiveLearningLoop
        Args:
            dataset : the current dataset that needs to be updated.
                It is assumed that it is the same object across the whole training
        """
        logger = logging.getLogger("activedet.engine.ActiveDatasetUpdater")

        if self.train_out_features:
            centers = self.extract_train_features(dataset, self.train_out_features)
            self.heuristic.center_features = centers

        pool = dataset.pool

        if len(pool) <= 0:
            return

        to_label = self.rank(pool)
        # Get the original index from the subset
        if self.ndata_to_label > len(to_label):
            logger.error(
                f"Unable to add {self.ndata_to_label}. Added {len(to_label)} only."
            )
        dataset.acquire(to_label[: self.ndata_to_label])

