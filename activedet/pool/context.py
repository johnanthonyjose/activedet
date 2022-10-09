from contextlib import contextmanager
import torch
import torch.nn as nn

from activedet.engine import TrainingMode, get_mode


@contextmanager
def trivial_context(model: nn.Module) -> None:
    """A context manager that does nothing
    """
    yield model


@contextmanager
def infer_uncertainty_context(model: nn.Module) -> None:
    """A context manager that temporarily sets 
    the mode of model into TrainingMode.INFER_UNCERTAINTY
    """
    assert model.training is False, "Ensure that the model is in inference mode"
    mode = get_mode(model)
    # Detach just to make sure, even if it's technically useless as we expect model to be in inference mode
    temp = mode.detach().clone()
    mode[0] = torch.tensor(TrainingMode.INFER_UNCERTAINTY,dtype=mode.dtype,device=mode.device)
    yield model
    mode[0] = temp[0]