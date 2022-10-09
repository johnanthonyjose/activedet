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
from torch.optim import swa_utils
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
from detectron2.data.samplers import InferenceSampler
from activedet.checkpoint import ActivedetCheckpointer
from activedet.data import ActiveDataset
from activedet.engine import hooks as activehooks
from activedet.engine.defaults import faster_setup
from activedet.engine.train_loop import ActiveTrainer
from activedet.pool import PoolRankStarter, ActiveDatasetUpdater
from activedet.solver import calculate_checkpoints
from activedet.utils.states import DetachMachinery
from activedet.utils.intermediate import get_attribute_recursive
from activedet.utils.model_ema import ModelEma

logger = logging.getLogger("activedet")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        results = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        plot = results.pop("plot",None)
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
    optim = instantiate(cfg.optimizer)
    checkpointables["optimizer"] = optim

    model_ema = None
    if cfg.train.model_ema.enabled is True:
        decay = cfg.train.model_ema.decay_rate
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: decay * averaged_model_parameter + (1 - decay) * model_parameter
        model_ema = swa_utils.AveragedModel(model,avg_fn=ema_avg)
        # model_ema = ModelEma(model,cfg.train.model_ema.decay_rate)
        checkpointables["online_state_dict"] = model_ema

    model = create_ddp_model(model, **cfg.train.ddp)

    def pool_builder(dataset):
        cfg.dataloader.pool.dataset = dataset
        return instantiate(cfg.dataloader.pool)

    def train_builder(dataset):
        cfg.dataloader.train.dataset = dataset
        return instantiate(cfg.dataloader.train)
    
    def train_one_epoch_builder(dataset):
        train_config = copy.deepcopy(cfg.dataloader.train)
        train_config.dataset = dataset
        train_config.sampler = InferenceSampler(len(dataset))
        return instantiate(train_config)


    cfg.active_learning.pool_evaluator.model = model_ema if model_ema else model
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
                start_checkpointer.resume_or_load(cfg.active_learning.start.init_checkpoint)
                cfg.active_learning.start.pool_evaluator.model = start_model
            cfg.active_learning.start.pool_evaluator.pool_builder = pool_builder
            start_pool_evaluator = instantiate(cfg.active_learning.start.pool_evaluator)

        dataset_starter = PoolRankStarter(pool_evaluator=start_pool_evaluator,
                                        heuristic=start_heuristic,
                                        start_n=cfg.active_learning.start_n,
                                        pool_transform=instantiate(cfg.active_learning.start_pool_transform),
                                        max_sample=cfg.active_learning.start.max_sample,
                                        seed=0,
                                        )
        cfg.dataloader.train.dataset.starter = dataset_starter
    train_loader = instantiate(cfg.dataloader.train)
    active_dataset = get_attribute_recursive(train_loader,ActiveDataset)
    trainer = ActiveTrainer(model, train_loader, optim)
    checkpointables["trainer"] = trainer
    checkpointables["active_dataset"] = active_dataset

    checkpoints, _ = calculate_checkpoints(
        start_n = cfg.active_learning.start_n,
        ims_per_batch = cfg.dataloader.train.batch_size,
        epoch_per_step = cfg.active_learning.epoch_per_step,
        ndata_to_label = cfg.active_learning.ndata_to_label,
        max_iter = cfg.train.max_iter,
        drop_start = cfg.active_learning.drop_start,
        drop_last = cfg.dataloader.train.drop_last
    )

    if cfg.train.detach_points:
        detacher = DetachMachinery(cfg.train.detach_points, checkpoints, last_iter=trainer.iter)
        checkpointables["detacher"] = detacher
    
    checkpointer = ActivedetCheckpointer(
        model,
        cfg.train.output_dir,
        **checkpointables
    )

    logger.info(f"Training restarts at the ff iterations: {checkpoints}")
    cfg.lr_multiplier.scheduler.checkpoints = checkpoints
    cfg.lr_multiplier.milestones.checkpoints = checkpoints
    cfg.lr_multiplier.warmup_lengths.checkpoints = checkpoints

    visualizer = instantiate(cfg.test.visualizer)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            # activehooks.ModelEmaHook(model_ema) if model_ema else None,
            activehooks.AveragedModelHook(checkpoints, model_ema,active_dataset,train_one_epoch_builder) if model_ema else None,
            activehooks.AperiodicCheckpointer(checkpointer,checkpoints, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            activehooks.AperiodicEvalHook(checkpoints, lambda: do_test(cfg, model_ema if model_ema else model))
            if cfg.test.with_evaluation
            else None,
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            activehooks.DatasetUpdaterHook(checkpoints,dataset_updater,active_dataset),
            activehooks.DropStartHook(checkpoints,active_dataset,checkpointer,cfg.train.init_checkpoint_path) 
            if cfg.active_learning.drop_start 
            else None,
            activehooks.DatasetVisualizerHook(checkpoints,visualizer,active_dataset)
            if cfg.test.is_visualize
            else None,
            activehooks.DataloaderRebuilderHook(checkpoints,active_dataset, train_builder),
            activehooks.ModelResetterHook(checkpoints,checkpointer, cfg.train.init_checkpoint_path),
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
        model_ema = None
        checkpointables = {}
        if cfg.train.model_ema.enabled is True:
            model_ema = ModelEma(model,cfg.train.model_ema.decay_rate)
            checkpointables["online_state_dict"] = model_ema

        model = create_ddp_model(model)
        ActivedetCheckpointer(model,**checkpointables).load(cfg.train.init_checkpoint)
        do_test(cfg, model_ema if model_ema else model)
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
