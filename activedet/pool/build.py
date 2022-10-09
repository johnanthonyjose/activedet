
from typing import Callable
import torch
import torch.nn as nn

from .dropout import MCdropout_context
from .context import trivial_context


def build_pool_context(cfg, model: nn.Module) -> Callable:
    """Builds the pool estimator context according to the config file
    Args:
        cfg (CfgNode): the all-in-one config file
        model (nn.Module): The model being trained
    Returns:
        a context manager that can modify the existing model in-place
        according to chosen estimator.
        It is expected to be used as either a context manager or part of ExitStack
        
    """
    estimator = None
    if cfg.ACTIVE_LEARNING.POOL.ESTIMATOR == "MCDropout":
        estimator = MCdropout_context
        assert sum(
            [
                isinstance(child, torch.nn.Dropout)
                or isinstance(child, torch.nn.Dropout2d)
                for child in model.modules()
            ]
        ), "MCDropout can only be used when the model has Dropout or Dropout2d module"
    elif cfg.ACTIVE_LEARNING.POOL.ESTIMATOR == "Trivial":
        estimator = trivial_context
    else:
        raise NotImplementedError(
            f"{cfg.ACTIVE_LEARNING.POOL.ESTIMATOR} is not available."
        )
    
    return estimator