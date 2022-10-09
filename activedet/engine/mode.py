from enum import IntFlag
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class TrainingMode(IntFlag):
    """Enumeration of allowable training mode for MI-AOD Implementation"""

    LABELED = 1
    """When it wants to train on labeled dataset similar to standard traning
    """

    UNLABELED = 2
    """When it wants to train on unlabeled dataset thru some metrics
    """

    MAX_UNCERTAINTY = 4
    """When it wants to train on unlabeled dataset in order to maximize instance uncertainty
    """

    MIN_UNCERTAINTY = 8
    """When it wants to train on unlabeled dataset in order to minimize instance uncertainty
    """
    
    INFER_UNCERTAINTY = 16
    """When it wants to do inference for instance uncertainty
    """

    INFER_BBOX = 32
    """When it wants to infer bounding boxes (Instances), which is the regular detection inference
    """


def get_mode(model: nn.Module):
    """Get the Model's mode that is expected to be stored in a buffer
    This mode contains the expected Training or Inference phase of the model
    Args:
        model (nn.Module): The model to be trained. In a single GPU, it is a
            regular nn.Module. In multi-GPU, it is wrapped around DDP.
    Returns:
        mode object registered as buffer in the model
    Note:
        In order to make it compatible on both single and multi-GPU. The
        implemententation is thru a recursive search.
    """
    module = model
    if isinstance(model, DistributedDataParallel):
        module = model.module
    modes = [m for name, m in module.named_buffers() if 'phase' == name]
    # We would assume that there's only one mode
    assert modes, f"The model {str(model)} does not have a mode!"
    return modes[0]