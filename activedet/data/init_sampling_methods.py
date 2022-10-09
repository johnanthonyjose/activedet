from typing import List, Optional
import torch
from torch.utils.data import Dataset


def balanced_look_ahead(dataset: Dataset, classlist: List[str], num_samples: int, heuristic: str, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Performs balanced look-ahead sampling to obtain initial dataset.

    Obtains an equal number of images per class. Images are classified as a whole based on a given heuristic.

    Args:
        dataset: Torch dataset to take samples from.
        classlist: List of classes available in the given dataset.
        num_samples: Total number of samples to obtain from the dataset.
        heuristic: Method for how an image is classified as a whole.
        generator: Optional; Generator used to ensure consistency during experimentation.

    Returns:
        A 1D torch.Tensor with size = len(dataset) containing integers corresponding to the the indices of the sampled images with respect to the dataset.
    """
    
    assert num_samples <= len(dataset), "Number of samples needed exceeds the length of the dataset"

    initials = torch.tensor([], dtype=torch.int64)
    is_sampled = torch.zeros(len(dataset), dtype=torch.bool)
    sampled_per_class = torch.zeros(len(classlist), dtype=torch.int64)

    num_per_class = int(num_samples / len(classlist))

    if heuristic == 'largest_area':
        sampling_criteria = largest_area
    else:
        raise NotImplementedError(f"No sampling heuristic named as {heuristic} is implemented.")

    while sampled_per_class.sum() < num_samples:
        start_sampled = sampled_per_class.sum()

        classes_to_sample = (sampled_per_class < num_per_class).nonzero().flatten()
        images_class = sampling_criteria(dataset, images_class, classes_to_sample)

        for class_id in classes_to_sample:
            num_to_sample = num_per_class - sampled_per_class[class_id]
            class_mask = (images_class == class_id) & ~ is_sampled
            class_subset = class_mask.nonzero().flatten()

            subset_inds = torch.randperm(len(class_subset), generator=generator)
            if len(class_subset) >= num_to_sample:
                subset_inds = subset_inds[:num_to_sample]
            else:
                subset_inds = subset_inds[:len(class_subset)]
            class_samples = class_subset[subset_inds]

            is_sampled[class_samples] = True
            sampled_per_class[class_id] += len(class_samples)
            initials = torch.cat((initials, class_samples), 0)

        assert start_sampled != sampled_per_class.sum(), f"No valid samples were obtained for the following classes: {classes_to_sample}"

    return initials


def largest_area(dataset, classlist: torch.Tensor) -> torch.Tensor:
    """Classifies a whole image based on the class of the largest object in the image.

    Args:
        dataset: Dataset where samples are obtained. This is referenced for the class of the objects.
        classlist: Tensor containing class_ids that lack samples. 
                   Class_ids with completed samples are ignored so that the total number of required samples is met.
    
    Returns:
        A 1D Tensor with size len(dataset), where each value pertains to the class of the image with the same index with reference to the dataset.
    """

    images_class = torch.empty(len(dataset), dtype=torch.int64)

    for index, sample in enumerate(dataset):
        objects_area = torch.empty(len(sample['annotations']))
        for ind, object in enumerate(sample['annotations']):
            area = (object['bbox'][2] - object['bbox'][0])*(object['bbox'][3] - object['bbox'][1])
            objects_area[ind] = area if object['category_id'] in classlist else -1

        largest_area = torch.argmax(objects_area)

        images_class[index] = sample['annotations'][largest_area]['category_id'] if objects_area[largest_area] > 0 else -1
    
    return images_class
