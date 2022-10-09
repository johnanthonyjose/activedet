from typing import Callable, Any, Dict, List
from collections import defaultdict
import logging
import itertools
import torch
import torch.nn as nn
import detectron2.utils.comm as comm
from detectron2.structures import Instances

from activedet.utils import divide

def parse_instance(instance: Instances, device="cpu") -> Instances:
    """Converts the detectron2 instances into activedet format
    Args:
        instances : prediction on a single image on one sample
    Returns:
        converted instances
    """
    # Hackish way of incorporating on both sem seg and box detection
    # Assumes that it's mutually exclusive to have both instances and sem_seg
    # It doesn't work on panoptic task
    try:
        parsed_instance = instance["sem_seg"]
    except KeyError:
        parsed_instance = instance.get("instances", None)
        # Assume that when it's none, it was already previously parsed
        if parsed_instance is None:
            parsed_instance = instance
    if device == "cpu":
        parsed_instance = parsed_instance.to(torch.device("cpu"))
    return parsed_instance

def parse_features(features:Dict[str, torch.Tensor], nodes: List[str],method="mean") -> torch.Tensor:
    """Converts each selected feature nodes into a vector.
    Args:
        features : sets of feature maps. It should have the same format as Detectron2's Resnet
        nodes: which of the feature maps are selected
        method: describes on how to reduce the a feature map into a vector.
            Available Options: [mean, max]
    """
    if method == "mean":
        pooler = nn.AdaptiveAvgPool2d((1,1))
    elif method == "max":
        pooler = nn.AdaptiveMaxPool2d((1,1))
    feats = [pooler(features[node]).squeeze() for node in nodes]
    feats = torch.cat(feats,dim=0).to(torch.device("cpu"))
    return feats

def parse_embedding(instance: Instances) -> torch.Tensor:
    """Extracts the detectron2 instances to collect the embedding
    Args:
        instances : prediction on a single image on one sample
    Returns:
        embedding vector
    """
    parsed = instance.get("instances", None)

    return parsed.pred_emb.squeeze().to(torch.device("cpu"))


class MapCollator:
    """Collates the processed images and predictions
    from multiple GPUs.

    Each GPUs will have their own version of the collator
    Typically, it is expected that it will be utilized inside a loop.
    Assumptions:
        - Size of dataset is already known beforehand
        - It utilizes all_gather method in distributed training.

    Attributes:
        sample_size : expected amount of monte carlo samples performed on each image
        batch_size : amount of images being inferred for each forward pass. Must change it
            depending on the number of GPUs used and type of GPU
        dataset_size : size of dataset being processed
        remaining_samples : amount of images in the dataset that still needs to be collected/completed
        collection : existing amount of samples collated per key
        output_preprocessor : function to be used for pre-processing

    """

    def __init__(
        self, sample_size: int, batch_size: int, output_preprocessor: Callable
    ) -> None:
        self._world_size = comm.get_world_size()
        self._rank = comm.get_rank()
        self.sample_size = sample_size
        self.batch_size = batch_size
        assert (
            batch_size % self._world_size == 0
        ), "Ensure that pool batch size is divisble by num gpus"
        self._dataset_size = 0
        self.remaining_samples = 0
        self.collection = defaultdict(list)
        self.preprocessor = output_preprocessor
        self.logger = logging.getLogger("activedet.pool.collator")
        pass

    @property
    def dataset_size(self):
        return self._dataset_size

    @dataset_size.setter
    def dataset_size(self, x):
        self._dataset_size = x
        self.remaining_samples = x

    def process(self, inputs, outputs):
        """Parses both input and output of the model prediction
        so that it will be further processed in downstream tasks

        Args:
            inputs (Dict[str, torch.Tensor]): Images in a batch
            outputs ([type]): [description]: Instances
        Notes:
            Each set of inputs might contain partial samples only
        """
        for image, pred in zip(inputs, outputs):
            self.collection[image["image_id"]].append(self.preprocessor(pred))
        if self.sample_size > 1:
            self.logger.info(
                f"rank {comm.get_local_rank()}: Collection Size: {len(self.collection)} each item: {[(key,len(values)) for key, values in self.collection.items()]}"
            )
        # else:
        #     self.logger.info(f"rank {comm.get_local_rank()}: Collection Size: {len(self.collection)}")

    @property
    def yieldable_keys(self):
        # In Monte Carlo Estimation, It is not guaranteed that each batch from
        # data loader would contain all repeated samples of each images.
        # Thus, it must be imperative that we would only yield when a datapoint
        # has completed the predictions for each of its monte carlo samples
        keys_ret = []
        for key, sample in self.collection.items():
            if len(sample) == self.sample_size:
                keys_ret.append(key)
            elif len(sample) > self.sample_size:
                # In a multi-gpu setting, if the total samples is not divisible number of workers,
                # the current workaround is to pad the samples to make it divisible.
                # Please see data/samplers/InferenceRepeatSamplers.py
                # Hence, in this evaluation, we remove the
                # TODO: remove padding
                # self.collection[key] = sample[:self.sample_size]
                self.logger.error(
                    f"More samples are found. on {key} with length {len(sample)}. Not yet tested!"
                )
                keys_ret.append(key)
        return keys_ret

    def ready(self) -> bool:
        """Assess whether it is already readily yielable

        Returns:
            bool: [description]
        """
        yieldable = len(self.yieldable_keys) >= self.batch_size
        yieldable_last = (self.remaining_samples < self.batch_size) and len(
            self.yieldable_keys
        ) == self.remaining_samples
        if yieldable_last:
            self.logger.info(
                f"The last incomplete batch with size {len(self.yieldable_keys)}"
            )

        return yieldable or yieldable_last

    def pop(self) -> Dict[str, Any]:
        """Collated output that is ready to be passed on.
        Each GPU will equally divide the batch size
        Returns
        """
        poopable_keys = self.yieldable_keys[: self.batch_size]
        # We first pop it from the collection.
        # But not all keys will be poppoed. The reason is because in multi-GPU training, you divide across all GPU
        # The main assumption is that ALL GPUs will have the identical collection.
        # But they only return a portion of it depending on their rank.
        all_returns = {key: self.collection.pop(key) for key in poopable_keys}
        # self.logger.info(f"rank {comm.get_local_rank()}: Collection Sent. Size: {len(self.collection)}")
        # Prioritize even distribution of these keys across multi-GPU
        # Even distribution is more important than complying with the batch size requirement.
        # Example #1: there are 8 samples 4 GPUs, then
        # rank      : 0   1   2   3
        # num sample: 2   2   2   2
        # Example #2: There are 6 samples 4 GPUs, then
        # rank      : 0   1   2   3
        # num sample: 2   2   1   1
        splits = divide(self._world_size, poopable_keys)
        # Get only the split that is applicable for the current GPU device
        target_split = itertools.islice(splits, self._rank, self._rank + 1)
        # Flatten the nested list [[1,2,3]] into [1,2,3]
        flattened_split = itertools.chain.from_iterable(target_split)

        # Reduce the remaining before returning
        self.remaining_samples -= self.batch_size

        ret = {key: all_returns[key] for key in flattened_split}
        return ret
