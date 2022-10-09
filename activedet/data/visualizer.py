from typing import List, Dict, Union
import warnings
from detectron2.config.config import configurable

import torch
from detectron2.utils.events import get_event_storage
from detectron2.data import MetadataCatalog


def get_area(boxes: torch.Tensor):
    """Calculates the area of the boxes
    Args:
        boxes = Nx4 format where N is the number of boxes
    Returns:
        area in Nx1 tensor
    """
    #      (     X2      -   X1      ) * (   Y2        -     Y1    )
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_aspect_ratio(boxes: torch.Tensor):
    """Calculates the aspect ratio of boxes
    Args:
        boxes = Nx4 format where N is the number of boxes
    Returns:
        aspect ratio in Nx1 tensor
    Notes:
        aspect_ratio = width / height
    Thus, a vertical rectangle has a ratio < 1
    While, a horizontal rectangle has a ratio > 1
    """
    #      (     X2      -   X1      ) / (   Y2        -     Y1    )
    return (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])


class DistributionVisualizer:
    """an object responsible for visualizing the data

    It will be useful for the visualization of histogram distribution in tensorbard

    Args:
        cfg : all-in-one config in detectron2 format
    Notes:
        add_histogram can only be invoked when it's inside the training loop.
        That is, the instantiated obejct must act as if it's a hook.
    """
    @configurable
    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names : defines the classes for all training set
        """

        self.class_names = class_names

    @classmethod
    def from_config(cls, cfg):
        if len(cfg['DATASETS']['TRAIN']) > 1:
            warnings.warn("DistributionVisualizer will use class names under the first of the train set")

        ret = {}
        # Assumes that there's only one training set in the train config.
        ret["class_names"] = MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]).thing_classes

        return ret


    def add_histogram(self, dataset: List[Dict[str, Union[str, int, List]]]) -> None:
        """Adds the distribution based on the current dataset
        Args:
            dataset = the current version of the dataset
        """
        instances = self.parse(dataset)
        num_classes = len(self.class_names)
        storage = get_event_storage()
        # Note: torch.histc only allows float
        storage.put_histogram("image/aspect ratio", instances["aspect_ratio"].float())
        storage.put_histogram("image/# objects per image", instances["num_object"].float())
        storage.put_histogram("image/avg object area per image", instances["avg_area"].float())
        storage.put_histogram("object/class distribution",instances['object_class'].float(),bins=num_classes)
        storage.put_histogram("object/area distribution",instances["object_area"].float())
        storage.put_histogram("object/aspect ratio distribution", instances["object_aspect_ratio"].float())

    def parse(self, dataset: List[Dict[str, Union[str, int, List]]]) -> Dict[str,torch.Tensor]:
        """Converts the detectron2-style annotation into a histogram
        Args:
            dataset = the current version of the dataset
        Returns:
            a dictionary of 1D tensors that contains the values for each target characteristics
        """
        dists = {
            "aspect_ratio": [],
            "num_object": [],
            "avg_area": [],
            "object_class": [],
            "object_area": [],
            "object_aspect_ratio": [],
        }
        for image in dataset:
            annos = image["annotations"]
            image_ratio = torch.as_tensor([image['width'] / image['height']])
            classes = torch.as_tensor(
                [x["category_id"] for x in annos if not x.get("iscrowd", 0)],
                dtype=torch.long,
            )
            areas = get_area(torch.Tensor([x["bbox"] for x in annos if not x.get("iscrowd", 0)]))
            aspect_ratios = get_aspect_ratio(torch.Tensor([x["bbox"] for x in annos if not x.get("iscrowd", 0)]))

            dists['aspect_ratio'].append(image_ratio)
            dists['num_object'].append(torch.as_tensor([len(annos)]))
            dists['object_class'].append(classes)
            dists['object_area'].append(areas)
            dists['avg_area'].append(areas[None,...].mean(dim=0))
            dists['object_aspect_ratio'].append(aspect_ratios)

        dists = {key: torch.cat(value) for key,value in dists.items()}

        return dists
        
