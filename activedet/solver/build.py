# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import logging
import torch

from fvcore.common.param_scheduler import (
    PolynomialDecayParamScheduler,
)
from detectron2.config import CfgNode
from detectron2.solver import LRMultiplier, build_lr_scheduler as build_d2_lr_scheduler

from .lr_scheduler import (
    WarmupPolyLR,
    MultiWarmupParamScheduler,
    ConstantParamScheduler,
)

from .param_scheduler import ActiveMultiStepParamScheduler, get_multi_warmup_lengths, get_milestones

def get_basic_lr_scheduler_params(model):
    param_groups = list(filter(lambda p: p.requires_grad, model.parameters()))
    param_groups = [{
        'params': param_groups
    },]
    return param_groups

def build_active_lr_scheduler(
    cfg: CfgNode, checkpoints: List[int], optimizer: torch.optim.Optimizer
) -> LRMultiplier:
    """Builds a learning rate scheduler that fits on multi-training regime such as Active Learning
    It restarts the Warmup on each chekpoint.
    Args:
        cfg = the all-in-one configuration
        checkpoints = a list of iteration where it would start a new training regime
        optimizer = the loss function used
    Returns:
        an LRMultiplier which describes the factor multiplied to BASE_LR for each iteration
        This LRMultiplier is defined across the whole active step. It expects a reset for each active step.
        Thereby, it also resets the LR Scheduler for every new step
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    max_iter = cfg.SOLVER.MAX_ITER
    milestones = get_milestones(checkpoints, max_iter)

    if name == "WarmupPolyLR":
        scheduler = PolynomialDecayParamScheduler(
            base_value=cfg.SOLVER.BASE_LR, power=cfg.SOLVER.POLY_LR_POWER
        )
    elif name == "Constant":
        scheduler = ConstantParamScheduler(value=1)
    elif name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]

        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        scheduler = ActiveMultiStepParamScheduler(checkpoints,steps,max_iter,cfg.SOLVER.GAMMA)
    else:
        raise ValueError(f"Unknown LR Scheduler: {name}")

    warmup_lengths = get_multi_warmup_lengths(
        checkpoints, cfg.SOLVER.WARMUP_ITERS, max_iter
    )

    scheduler = MultiWarmupParamScheduler(
        scheduler,
        milestones,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_lengths=warmup_lengths,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    return LRMultiplier(optimizer, multiplier=scheduler, max_iter=cfg.SOLVER.MAX_ITER)


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POLY_LR_POWER,
            constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
        )
    else:
        return build_d2_lr_scheduler(cfg, optimizer)
