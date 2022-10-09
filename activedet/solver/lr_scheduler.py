# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List
from itertools import chain
import torch

from fvcore.common.param_scheduler import CompositeParamScheduler, LinearParamScheduler, ParamScheduler
from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

class ConstantParamScheduler(ParamScheduler):
    """
    Returns a constant value for a optimizer param.
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, where: float) -> float:
        if (where - self.WHERE_EPSILON) >= 1.0:
            raise RuntimeError(f"Invalid where parameter for scheduler: {where}")
        return self._value

class MultiWarmupParamScheduler(CompositeParamScheduler):
    """Adds an initial warmup stage for each step
    It treats each milestone as a new training regime.

    Args:
        scheduler = scheduler algorithm to use after the initial warmup
        milestones = the set of steps wherein it will restart the warmup
        warmup_factor = multipler to the intial value set by `solver.base_lr`
        warmup_lengths = relative lengths of the warmup for all step
        warmup_method = "linear" or "constant"
    
    Example:
        milestones = [0.2, 0.4, 0.6,0.8]

        Meaning, it will perform warmup five times, during: 0,0.2,0.4,0.6,0.8
    """
    def __init__(self,
                scheduler: ParamScheduler, 
                milestones: List[float],
                warmup_factor: float, 
                warmup_lengths: List[float],
                warmup_method: str = "linear"
                ):
        if milestones:
            end_warmualue = scheduler(warmup_lengths[0] * milestones[0])
        else:
            end_warmualue = scheduler(warmup_lengths[0] * 1.0)
        start_warmvalue = warmup_factor * scheduler(0.0)
        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_warmvalue)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_warmvalue, end_warmualue)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))

        # +1 is introduced for the initial random-based training
        schedulers = [warmup, scheduler] * (len(milestones) + 1)
        # LR Scheduler is fixed because Multiwarmup is designed for the whole training step
        # Thus, no need to rescaled it.
        interval_scaling = ["rescaled", "fixed"] * (len(milestones) + 1)
        total = [0] + milestones + [1]
        
        def step_length(warmup_length, length):
            warmup_size = warmup_length * length
            return (warmup_size, length-warmup_size)
        # On 15 active step, there are 16 training regime (due to the addition of random initial training)
        def difference(total):
            return [b-a for a,b in zip(total[:-1],total[1:])]

        #Still not working
        lengths = list(chain.from_iterable(step_length(warmup, diff) for warmup, diff in zip(warmup_lengths, difference(total))))

        super().__init__(
            schedulers,
            interval_scaling=interval_scaling,
            lengths=lengths
            )

        


# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()