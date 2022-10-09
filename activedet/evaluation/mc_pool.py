from typing import Callable, Generator, Dict, List
import copy
import itertools
from functools import partial
from contextlib import ExitStack, contextmanager
import logging
import time
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from detectron2.config import CfgNode, configurable
from detectron2.evaluation import inference_context
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data.dataset_mapper import DatasetMapper

from activedet.data.build import build_detection_pool_loader
from activedet.pool import build_pool_context, MapCollator, parse_instance


class MCEvaluator:
    """Evaluates the Pool through Monte Carlo Estimation in a paralled fashion
    Unlike the regular detectron2 inference, It works well on multi-gpu setting

    Attributes:
        model (nn.Module): Describes the model used for evaluation
        build_pool_loader (Callable): A function that builds the dataloader. It takes dataset as its argument
        pool_context(contextmanager): Dynamically changes the model training mode. It takes model as input
        collator (MapCollator): It processes the input & output pairs of each forward pass so that all
            MonteCarlo samples are combined together in one key.
        logger (logging.Logger): logging object used for debugging

    Args:
        sample_size: Amount of repeats an image would have. 
            Set it to 1 in order not to have repeats
        pool_batch_size: Batch size for pool data loader
        pool_context(contextmanager): Dynamically changes the model training mode. It takes model as input
        model: Describes the model used for evaluation
        pool_builder (Callable): builds pool data loader once called. It takes dataset as input and returns a dataloader
        collate_processor (Callable): Describes on how to parse instances on every (batch) prediction
        target_model : A filler argument

    Notes:
        In case you don't need a monte carlo sampling, just set cfg.ACTIVE_LEARNING.POOL.SAMPLING_METHOD into Trivial

    """

    @configurable
    def __init__(
        self,
        *,
        sample_size: int,
        pool_batch_size: int,
        pool_context: contextmanager,
        model: nn.Module,
        pool_builder: Callable,
        collate_processor=parse_instance,
        target_model: nn.Module = None,
    ):
        assert (
            pool_batch_size % comm.get_world_size() == 0
        ), "Pool batch size must be divisble by the number of GPUs"
        self.build_pool_loader = pool_builder
        self.pool_context = pool_context
        self.collator = MapCollator(sample_size, pool_batch_size, collate_processor)
        self.sample_size = int(sample_size)
        self.model = model

        assert (
            self.sample_size >= 1
        ), "Sample size must be a positive integer not {self.sample_size}"

    @classmethod
    def from_config(cls, cfg, model):
        ret = {}
        ret["sample_size"] = (
            cfg.ACTIVE_LEARNING.POOL.MC_SIZE
            if cfg.ACTIVE_LEARNING.POOL.SAMPLING_METHOD == "MonteCarlo"
            else 1
        )
        ret["pool_batch_size"] = cfg.ACTIVE_LEARNING.POOL.BATCH_SIZE
        ret["pool_context"] = build_pool_context(cfg, model)
        ret["model"] = model
        ret["pool_builder"] = partial(build_detection_pool_loader, cfg=cfg)
        return ret

    def __call__(
        self, pool: torch.utils.data.Dataset
    ) -> Generator[Dict[str, List[torch.Tensor]], None, None]:
        """Evaluates the pool dataset

        Args:
            pool (torch.utils.data.Dataset): An unlabeled dataset to be evaluated

        Yields:
            Generator[Dict[str, List[torch.Tensor]], None, None]: lazily loads and perform the pool inference
                Yields a batch as defined by pool loader
                Example:
                    num gpus = 4
                    pool loader batch = 8
                    MC Dropout sample = 10
                    Each GPU yields 2 samples wherein each sample has size 10

        Notes:
            It uses yield rather than return.
            That is, this is a generator function rather than regular function
        """
        num_devices = comm.get_world_size()
        logger = logging.getLogger("activedet.evaluation.mc_pool")
        loader = self.build_pool_loader(dataset=pool)
        total = len(loader)
        logger.info("Start Pool inference on {} batches".format(total))

        logger.info(
            f"The first 10 images in pool are {[pool[i]['image_id'] for i in range(10)]}"
        )

        self.collator.dataset_size = len(pool)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        total_heuristic_time = 0
        with ExitStack() as stack:
            stack.enter_context(inference_context(self.model))
            estimator = stack.enter_context(self.pool_context(self.model))
            stack.enter_context(torch.no_grad())
            start_data_time = time.perf_counter()
            for idx, images in enumerate(loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                pred = estimator(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                if self.sample_size > 1:
                    # Gather images and predictions among all workers to all workers
                    all_pred = comm.all_gather(pred)
                    all_images = comm.all_gather(images)
                    # When results are gathered from multiple GPU, it results to a list
                    # wherein each index represents the cuda device
                    # Flatten the list of instance list
                    preds = itertools.chain.from_iterable(all_pred)
                    ims = itertools.chain.from_iterable(all_images)
                else:
                    preds = pred
                    ims = images
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                self.collator.process(ims, preds)
                total_eval_time += time.perf_counter() - start_eval_time

                start_heuristic_time = time.perf_counter()
                if self.collator.ready():
                    poops = self.collator.pop()

                    if self.sample_size > 1:
                        all_poops = comm.gather(poops)
                        for i, poop in enumerate(all_poops):
                            logger.info(
                                f"rank {i}: yields {[p for p in poop]} keys with sizes {[len(poop[k]) for k in poop]}"
                            )
                    yield poops
                total_heuristic_time += time.perf_counter() - start_heuristic_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                heuristics_seconds_per_iter = total_heuristic_time / iters_after_start
                total_seconds_per_iter = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(
                        seconds=int(total_seconds_per_iter * (total - idx - 1))
                    )
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Heuristic: {heuristics_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(
            datetime.timedelta(seconds=int(total_compute_time))
        )
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str,
                total_compute_time / (total - num_warmup),
                num_devices,
            )
        )
        return


class GNEvaluator:
    """Gaussion Noise Evaluator

    Adds gaussian noise of different levels for the input images

    This is to be used in conjuction with the Kao_LS (Localization Stability) heursitic
    """

    def __init__(self, cfg: CfgNode, model: nn.Module):
        self.model = model
        self.pool_batch_size = cfg.ACTIVE_LEARNING.POOL.BATCH_SIZE
        self.mapper = DatasetMapper(cfg, False)
        self.sd_levels = cfg.ACTIVE_LEARNING.POOL.SD_LEVELS
        self.build_pool_loader = partial(build_detection_pool_loader, cfg=cfg)

        self.stochastic_model = copy.deepcopy(model)
        self.stochastic_model.eval()

    def __call__(
        self, dataset: torch.utils.data.Dataset
    ) -> Generator[Dict[str, List[torch.Tensor]], None, None]:
        loader = self.build_pool_loader(dataset=dataset)
        isTrain = self.model.training
        if isTrain:
            self.model.eval()

        # Is this trick gonna work on multi-GPU?
        device = next(self.model.parameters()).device
        self.model = self.model.to(torch.device("cpu"))
        self.stochastic_model.load_state_dict(self.model.state_dict())
        self.stochastic_model.to(device)
        self.stochastic_model.eval()

        # output = {}
        for idx, images in enumerate(tqdm(loader, total=len(loader))):
            with torch.no_grad():
                # Repeat images for monte carlo
                # e.g. original size = 2x3x224x224
                #      order (dim 0) = [A B]
                # after repeat = 60x3x224x224
                #       order (dim 0) = [A A A A .. A A B B B B B .. B B]
                # TODO: Put repeat inside the dataloader.
                # For now, hack first
                data = []

                for instance in images:
                    for sd in self.sd_levels:
                        data.append(self.add_gaussian_noise(instance, sd))

                # data has a very big batch_size := images.batch_size x sample_size
                # Iterate first over
                # TODO: Look for possible vectorization
                output = {}
                for data_chunk in self.chunks(data, self.pool_batch_size):
                    # Infer as a batch
                    pred = self.stochastic_model(
                        data_chunk
                    )  # dims: (pool_batch_size)xCxHxW

                    # Process the prediction individually
                    # Due to MCDropout, it's possible that the current pool batch doesn't have
                    # all the samples for a single image
                    # An individual loop helps organizes each MCDropout samples as one key
                    for image, instance in zip(data_chunk, pred):
                        preds = output.get(image["image_id"], [])

                        # Hackish way of incorporating on both sem seg and box detection
                        # Assumes that it's mutually exclusive to have both instances and sem_seg
                        # It doesn't work on panoptic task
                        try:
                            instance = instance["sem_seg"]
                        except KeyError:
                            instance = instance.get("instances", None)

                        preds.append(instance.to(torch.device("cpu")))
                        output[image["image_id"]] = preds

                        del preds
                        del instance
                        del image
                    del pred
                    del data_chunk

                # Uses yield rather than return
                # Creates a generator function rather than regular function
                # It lazily loads and perform the pool inference
                # Yiels a batch as defined by pool loader
                # E.g.
                #   pool loader batch = 4
                #   MC Dropout sample = 10
                # Yiels: a batch of 40
                yield output

        self.model.train(isTrain)
        self.stochastic_model.to(torch.device("cpu"))
        self.model.to(device)

    def add_gaussian_noise(self, image_instance, sd):
        image_instance_copy = copy.deepcopy(image_instance)
        input_im = image_instance_copy["image"]
        input_im = input_im + sd * torch.randn(input_im.size())

        # noisy_image = noisy_image.to(torch.uint8)
        input_im = torch.clamp(input_im, 0, 255).to(dtype=torch.uint8)  # with clamping

        image_instance_copy["image"] = input_im

        # return image_instance # the returned image is same when using this return
        # TODO: fix the memory management
        return image_instance_copy

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

        del lst


class BBEvaluator:
    """Backbone Evaluator

    returns the backbone FPN features

    To be used with coreset approach
    """

    def __init__(self, cfg: CfgNode, model: nn.Module):
        self.model = model
        self.pool_batch_size = cfg.ACTIVE_LEARNING.POOL.BATCH_SIZE
        self.mapper = DatasetMapper(cfg, False)
        self.build_pool_loader = partial(build_detection_pool_loader, cfg=cfg)
        self.in_feature = cfg.ACTIVE_LEARNING.CORESET.IN_FEATURE
        self.feature_height = cfg.ACTIVE_LEARNING.CORESET.FEATURE_HEIGHT
        self.feature_width = cfg.ACTIVE_LEARNING.CORESET.FEATURE_WIDTH

        assert (
            cfg.ACTIVE_LEARNING.POOL.ESTIMATOR == "Trivial"
        ), "Trivial evaluator must be used"

        self.eval_model = copy.deepcopy(model)
        self.eval_model.eval()

    def __call__(
        self, dataset: torch.utils.data.Dataset
    ) -> Generator[Dict[str, List[torch.Tensor]], None, None]:
        loader = self.build_pool_loader(dataset=dataset)
        isTrain = self.model.training
        if isTrain:
            self.model.eval()

        # Is this trick gonna work on multi-GPU?
        device = next(self.model.parameters()).device
        self.model = self.model.to(torch.device("cpu"))
        self.eval_model.load_state_dict(self.model.state_dict())
        self.eval_model.to(device)
        self.eval_model.eval()

        # output = {}
        for idx, images in enumerate(tqdm(loader, total=len(loader))):
            with torch.no_grad():

                output = {}
                output_instances = {}
                output_features = {}
                for d_idx, data_chunk in enumerate(
                    self.chunks(images, self.pool_batch_size)
                ):
                    # Infer as a batch

                    input_images = self.eval_model.preprocess_image(
                        data_chunk
                    )  # preprocess batched image
                    batch_features = self.eval_model.backbone(input_images.tensor)[
                        self.in_feature
                    ]  # forward pass the backbone

                    avg_pool = nn.AdaptiveAvgPool2d(
                        (self.feature_height, self.feature_width)
                    )  # output size is 15 by 15, this is picked arbitrarily

                    batch_features = avg_pool(
                        batch_features
                    )  # make output size N x C x 15 x 15,
                    # so that all image sizes will have the same size of feature vectors

                    batch_features = torch.flatten(
                        batch_features, start_dim=1
                    )  # flatten

                    for image, features in zip(data_chunk, batch_features):
                        output_features[image["image_id"]] = features.to(
                            torch.device("cpu")
                        )

                        del image
                        del features

                    del input_images
                    del batch_features
                    del data_chunk

                output = {"instances": output_instances, "features": output_features}
                yield output

        self.model.train(isTrain)
        self.eval_model.to(torch.device("cpu"))
        self.model.to(device)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

        del lst


def build_pool_evaluator(cfg, model):
    if cfg.ACTIVE_LEARNING.POOL.EVALUATOR == "MCEvaluator":
        return MCEvaluator(cfg, model)
    elif cfg.ACTIVE_LEARNING.POOL.EVALUATOR == "GNEvaluator":
        return GNEvaluator(cfg, model)
    elif cfg.ACTIVE_LEARNING.POOL.EVALUATOR == "BBEvaluator":
        return BBEvaluator(cfg, model)
    else:
        raise NotImplementedError(
            f"{cfg.ACTIVE_LEARNING.POOL.EVALUATOR} is not implemented"
        )
