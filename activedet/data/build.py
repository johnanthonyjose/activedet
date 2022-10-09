
import logging
import random
import operator
import numpy

import torch
from detectron2.config import configurable
from detectron2.data.common import DatasetFromList, MapDataset, AspectRatioGroupedDataset
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage
from detectron2.data.build import (
    get_detection_dataset_dicts,
    trivial_batch_collator,
    worker_init_reset_seed
)
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.utils.comm as comm
from detectron2.data.samplers import TrainingSampler, InferenceSampler, RepeatFactorTrainingSampler

from activedet.data.samplers import InferenceRepeatSampler, TrivialSampler

def build_data_loader(dataset, batch_size: int, num_workers: int, training=True, drop_last=False):
    """Builds Data Loader for Image Classification Task
    Specifically, it doesn't require the MapDataset and DataMapper in order to simplify it
    However, dataset object must return the item in Detectron2 format.
    Please see `activedet.data.datasets.CIFAR` for example
    Args:
        dataset(torch.utils.data.Dataset): dataset object
    """
    world_size = get_world_size()
    assert (
        batch_size > 0 and batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        batch_size, world_size
    )
    mini_batch_size = batch_size // world_size
    sampler = (TrainingSampler if training else InferenceSampler)(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, mini_batch_size, drop_last=drop_last
        )
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed
    )

def build_pool_loader(dataset,*, batch_size, num_workers,sampling_method="Trivial",num_repeats=1,seed=None):
    """Pool Data Loader for Image Classification Task
    """
    world_size = get_world_size()
    assert (
        batch_size > 0 and batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        batch_size, world_size
    )
    logger = logging.getLogger(__name__)
    if sampling_method == "MonteCarlo":
        num_repeats = num_repeats
        sampler = InferenceRepeatSampler(len(dataset),num_repeats=num_repeats)
        logger.info("Using Pool sampler InferenceRepeatSampler")
    elif sampling_method == "Trivial":
        sampler = TrivialSampler(len(dataset),batch_size)
        logger.info("Using Pool sampler TrivialSampler")
    else:
        raise ValueError("Unknown sampling method: {}".format(sampling_method))


    mini_batch_size = batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, mini_batch_size, drop_last=False
    )
    if seed is None:
        seed = comm.shared_random_seed() #Ensure that all workers have the same seed for deterministic output

    g = torch.Generator()
    g.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,  # Trivial collate fn
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker # Reproduce dataset sampling across all workers
    )

def _pool_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            [dataset],
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset)]
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)

    total_batch_size = cfg.ACTIVE_LEARNING.POOL.BATCH_SIZE

    if sampler is None:
        sampler = cfg.ACTIVE_LEARNING.POOL.SAMPLING_METHOD

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "batch_size": total_batch_size,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "seed": cfg.SEED,
        "num_repeats": cfg.ACTIVE_LEARNING.POOL.MC_SIZE
    }


@configurable(from_config=_pool_loader_from_config)
def build_detection_pool_loader(dataset, *, mapper, sampler, batch_size, num_workers=0, seed=None, num_repeats=1):
    """
    Similar to `detectron2.build_detection_test_loader`.
    It doesn't contain any configurable as it won't be loaded from cfg file.
    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or str): a sampler that produces
            indices to be applied on ``dataset``.
        batch_size (int): total batch size across all workers.
        num_workers (int): number of parallel data loading workers
        seed (int): initial seed
        num_repeats(int): For Monte Carlo Sampling, number of repeats to be performed on a given dataset
    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    Examples:
    ::
        data_loader = build_detection_pool_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))
    Notes:

    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    world_size = get_world_size()
    assert (
        batch_size > 0 and batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        batch_size, world_size
    )
    if isinstance(sampler,str):
        logger = logging.getLogger(__name__)
        if sampler == "MonteCarlo":
            sampler = InferenceRepeatSampler(len(dataset),num_repeats=num_repeats)
            logger.info("Using Pool sampler InferenceRepeatSampler")
        elif sampler == "Trivial":
            sampler = TrivialSampler(len(dataset),batch_size)
            logger.info("Using Pool sampler TrivialSampler")
        elif sampler == "TrainingSampler":
            # For unsupervised learning
            sampler = TrainingSampler(len(dataset))
            logger.info("Using Pool sampler TrainingSampler")
        else:
            raise ValueError("Unknown sampling method: {}".format(sampler))

    minibatch_size = batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, minibatch_size, drop_last=False
    )
    if seed is None:
        seed = comm.shared_random_seed() #Ensure that all workers have the same seed for deterministic output

    g = torch.Generator()
    g.manual_seed(seed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,  # Trivial collate fn
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker # Reproduce dataset sampling across all workers
    )
    return data_loader

def seed_worker(worker_id):
    """Ensures that all workers would start at the same seed.
    It uses for monte carlo sampling
    Args:
        worker_id (int): worker id (rank)
    """
    worker_seed = comm.shared_random_seed()
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def _val_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TEST,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset)]
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    if mapper is None:
        # Setting is_train to True means that ground truth remains intact
        # TODO: Create new Mapper. Logs shows training rather than validation.
        mapper = DatasetMapper(cfg, is_train=True)

    if sampler is None:
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "mapper": mapper,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }

@configurable(from_config=_val_loader_from_config)
def build_detection_val_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    num_workers=0
):
    """ Similar to `build_detection_train_loader` but we change the default dataset and mapper
    """
    return build_detection_train_loader(
        dataset,
        mapper=mapper,
        sampler=sampler,
        total_batch_size=total_batch_size,
        aspect_ratio_grouping=False,
        num_workers=num_workers,
    )

def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0, drop_last=False
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            drop_last=drop_last,
            pin_memory=True,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=drop_last
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
            pin_memory=True,
        )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "drop_last": cfg.DATALOADER.DROP_LAST
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, batch_size, aspect_ratio_grouping=True, num_workers=0, drop_last=False
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        drop_last=drop_last,
    )
