#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import copy
import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from activedet.checkpoint import ActivedetCheckpointer
from activedet.data import ActiveDataset
from activedet.engine import hooks as activehooks, get_mode
from activedet.engine.defaults import faster_setup
from activedet.engine.mode import TrainingMode
from activedet.engine.train_loop import SemiSupervisedTrainer
from activedet.pool import PoolRankStarter, ActiveDatasetUpdater
from activedet.solver import calculate_checkpoints
from activedet.utils.states import DetachMachinery, MIAODTrainingPhase
from activedet.utils.intermediate import get_attribute_recursive

logger = logging.getLogger("activedet")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        mode = get_mode(model)
        mode[0] = torch.tensor(
            TrainingMode.INFER_BBOX, dtype=mode.dtype, device=mode.device
        )
        # Ensure that all replicas will have the same attribute
        comm.synchronize()
        results = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        plot = results.pop("plot", None)
        print_csv_format(results)
        if plot is not None:
            results["plot"] = plot
        return results


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("activedet")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    checkpointables = {}
    cfg.optimizer.params.model = model
    def optim_callback():
        return instantiate(cfg.optimizer.params)

    optim = instantiate(cfg.optimizer)
    checkpointables["optimizer"] = optim

    model = create_ddp_model(model, **cfg.train.ddp)

    def pool_builder(dataset):
        cfg.dataloader.pool.dataset = dataset
        return instantiate(cfg.dataloader.pool)

    def unsupervised_builder(dataset):
        untrain_cfg = copy.deepcopy(cfg.dataloader.train)
        untrain_cfg.dataset = dataset
        untrain_cfg.mapper.is_train = False
        return instantiate(untrain_cfg)

    def train_builder(dataset):
        cfg.dataloader.train.dataset = dataset
        return instantiate(cfg.dataloader.train)

    cfg.active_learning.pool_evaluator.model = model
    cfg.active_learning.pool_evaluator.pool_builder = pool_builder
    pool_evaluator = instantiate(cfg.active_learning.pool_evaluator)
    heuristic = instantiate(cfg.active_learning.heuristic)
    dataset_updater = ActiveDatasetUpdater(
        pool_evaluator=pool_evaluator,
        heuristic=heuristic,
        ndata_to_label=cfg.active_learning.ndata_to_label,
        max_sample=cfg.active_learning.pool.max_sample,
        train_out_features=cfg.active_learning.train_out_features,
        seed=0,
    )

    if cfg.active_learning.start.heuristic:
        start_heuristic = instantiate(cfg.active_learning.start.heuristic)

        start_pool_evaluator = pool_evaluator
        if cfg.active_learning.start.pool_evaluator:
            cfg.active_learning.start.pool_evaluator.model = model
            if cfg.active_learning.start.model:
                start_model = instantiate(cfg.active_learning.start.model)
                start_checkpointer = ActivedetCheckpointer(start_model)
                start_checkpointer.resume_or_load(
                    cfg.active_learning.start.init_checkpoint
                )
                cfg.active_learning.start.pool_evaluator.model = start_model
            cfg.active_learning.start.pool_evaluator.pool_builder = pool_builder
            start_pool_evaluator = instantiate(cfg.active_learning.start.pool_evaluator)

        dataset_starter = PoolRankStarter(
            pool_evaluator=start_pool_evaluator,
            heuristic=start_heuristic,
            start_n=cfg.active_learning.start_n,
            pool_transform=instantiate(cfg.active_learning.start_pool_transform),
            max_sample=cfg.active_learning.start.max_sample,
            seed=0,
        )
        cfg.dataloader.train.dataset.starter = dataset_starter
    train_loader = instantiate(cfg.dataloader.train)
    active_dataset = get_attribute_recursive(train_loader, ActiveDataset)
    unlabeled_loader = unsupervised_builder(active_dataset.pool)
    trainer = SemiSupervisedTrainer(
        model,
        train_loader,
        optim,
        unlabeled_loader=unlabeled_loader,
        amp_enabled=cfg.train.amp.enabled,
        optim_resetter=optim_callback,
    )
    checkpointables["trainer"] = trainer
    checkpointables["active_dataset"] = active_dataset

    checkpoints, _ = calculate_checkpoints(
        start_n=cfg.active_learning.start_n,
        ims_per_batch=cfg.dataloader.train.batch_size,
        epoch_per_step=cfg.active_learning.epoch_per_step,
        ndata_to_label=cfg.active_learning.ndata_to_label,
        max_iter=cfg.train.max_iter,
        drop_start=cfg.active_learning.drop_start,
        drop_last=cfg.dataloader.train.drop_last,
    )

    max_points = getattr(cfg.train, "max_uncertainty_points", [])
    min_points = getattr(cfg.train, "min_uncertainty_points", [])
    if min_points or max_points:
        phase_watcher = MIAODTrainingPhase(
            cfg.train.labeled_points, min_points, max_points
        )
        checkpointables["phase_watcher"] = phase_watcher

    if cfg.train.detach_points:
        detacher = DetachMachinery(
            cfg.train.detach_points, checkpoints, last_iter=trainer.iter
        )
        checkpointables["detacher"] = detacher

    checkpointer = ActivedetCheckpointer(model, cfg.train.output_dir, **checkpointables)

    logger.info(f"Training restarts at the ff iterations: {checkpoints}")
    cfg.lr_multiplier.scheduler.checkpoints = checkpoints
    cfg.lr_multiplier.milestones.checkpoints = checkpoints
    cfg.lr_multiplier.warmup_lengths.checkpoints = checkpoints

    visualizer = instantiate(cfg.test.visualizer)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            activehooks.AperiodicEvalHook(checkpoints, lambda: do_test(cfg, model))
            if cfg.test.with_evaluation
            else None,
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            activehooks.TrainingPhaseHook(watcher=phase_watcher)
            if min_points or max_points
            else None,
            activehooks.DatasetUpdaterHook(
                checkpoints, dataset_updater, active_dataset
            ),
            activehooks.DropStartHook(
                checkpoints,
                active_dataset,
                checkpointer,
                cfg.train.init_checkpoint_path,
            )
            if cfg.active_learning.drop_start
            else None,
            activehooks.AperiodicCheckpointer(
                checkpointer, checkpoints, **cfg.train.checkpointer
            )
            if comm.is_main_process()
            else None,
            activehooks.DatasetVisualizerHook(checkpoints, visualizer, active_dataset)
            if cfg.test.is_visualize
            else None,
            activehooks.DataloaderRebuilderHook(
                checkpoints, active_dataset, train_builder, kind="labeled"
            ),
            activehooks.DataloaderRebuilderHook(
                checkpoints, active_dataset.pool, unsupervised_builder, kind="unlabeled"
            ),
            activehooks.ModelResetterHook(
                checkpoints, checkpointer, cfg.train.init_checkpoint_path
            ),
            activehooks.DetachModuleHook(node=cfg.train.detach_node, detacher=detacher)
            if cfg.train.detach_points
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
        # Save initial pretrained weights for model resets
        checkpointer.save(cfg.train.init_checkpoint_path)

    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    faster_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        ActivedetCheckpointer(model).load(cfg.train.init_checkpoint)
        do_test(cfg, model)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
