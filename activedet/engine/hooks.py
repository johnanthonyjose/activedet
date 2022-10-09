from contextlib import ExitStack
from typing import Callable, Optional, List, Any, Dict
import logging
import os
import re
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel
from iopath.common.file_io import PathManager
from fvcore.common.checkpoint import Checkpointer
from detectron2.engine import HookBase
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm
from detectron2.utils.env import TORCH_VERSION

from activedet.data import ActiveDataset
from activedet.engine import TrainingMode
from activedet.evaluation.evaluator import validation_context
from activedet.utils.states import DetachMachinery, MIAODTrainingPhase
from activedet.utils.intermediate_layer import get_child_module


def unwrap_module(model):
    original = model
    if isinstance(model, DistributedDataParallel):
        original = model.module

    return original



class ModelEmaHook(HookBase):
    """Create Smooth version of the weights of the training model
    This is inspired from Tensorflow and MoCO
    """
    def __init__(self, model_ema) -> None:
        super().__init__()
        self.model_ema = model_ema
        
    def after_step(self):
        model = unwrap_module(self.trainer.model)
        self.model_ema.update(model)


class DetachModuleHook(HookBase):
    """Hook wrapper for DetachMachinery
    Provides an ability to use DetachMachinery in Detectron2 hook-based system
    Attributes:
        detacher(DetachMachinery): respondible for detaching a module according to the break points
        node(str) : indicates which module within the model would need to be detached during certain periods in training
    Args:
        (node, detacher): same as attributes
    """
    def __init__(self, node: str, detacher: DetachMachinery):
        self.detacher = detacher
        self.node = self.parse_nodes(node)
    
    def parse_nodes(self, node):
        return re.sub("^model.","",node)

    def before_train(self):
        child = get_child_module(self.trainer.model, self.node)
        child.detacher = self.detacher

    def after_step(self):
        self.detacher.step()


class TrainingPhaseHook(HookBase):
    """Hook wrapper for MIAODTrainingPhase objects
    Provides an ability to use MIAODTrainingPhase in Detectron2 hook-based system
    """
    def __init__(self, watcher: MIAODTrainingPhase) -> None:
        self.watcher = watcher
        self.state_map = {
            'labeled': TrainingMode.LABELED,
            'min_uncertainty': TrainingMode.MIN_UNCERTAINTY,
            'max_uncertainty': TrainingMode.MAX_UNCERTAINTY,
        }
    
    def after_step(self):
        self.watcher.step()

        self.trainer.mode = self.state_map[self.watcher.state]



class IrregularCheckpointer:
    """
    Save checkpoints. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, only iteration is in checkpoints
    or if `max_iter` is reached.
    Attributes:
        checkpointer (Checkpointer): the underlying checkpointer object
        checkpoints (List[int]): list of iterations where the models is triggered to save
        max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "{file_prefix}_final" will be saved.
        max_to_keep (int): maximum number of most current checkpoints to keep,
            previous checkpoints will be deleted
        file_prefix (str): the prefix of checkpoint's filename
    """
    def __init__(
        self,
        checkpointer: Checkpointer,
        checkpoints: List[int],
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
    ) -> None:
        """
        Args:
            checkpointer: the checkpointer object used to save checkpoints.
            checkpoints (List[int]): list of iterations where the models is triggered to save
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "{file_prefix}_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
            file_prefix (str): the prefix of checkpoint's filename
        """
        self.checkpointer = checkpointer
        self.checkpoints = [int(iteration) for iteration in checkpoints]
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints: List[str] = []
        self.path_manager: PathManager = checkpointer.path_manager
        self.file_prefix = file_prefix

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.
        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) in self.checkpoints:
            self.checkpointer.save(
                "{}_{:07d}".format(self.file_prefix, iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                # pyre-fixme[58]: `>` is not supported for operand types `int` and
                #  `Optional[int]`.
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            # pyre-fixme[58]
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)
    
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.
        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)
    


class AperiodicCheckpointer(IrregularCheckpointer, HookBase):
    """
    Same as :class:`IrregularCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.
    It is executed at every `checkpoint` and after the last iteration.
    """
    
    def before_train(self):
        self.max_iter = self.trainer.max_iter
    
    def after_step(self):
        self.step(self.trainer.iter)


class MilestoneHook(HookBase):
    """Milestone Hooks are only triggered on certain iterations that is 
    defined in milestones. In contrast to Hooks, hooks are triggered every iterations

    Args:
        milestones (List[int]) : list of iterations on where this hook will be triggered
    """
    def __init__(self, milestones):
        super().__init__()
        self._milestones = milestones
    
    def after_step(self):
        if (self.trainer.iter + 1) in self._milestones:
            self.after_milestone()

    def after_milestone(self):
        """Defines the functionality to be triggered when encountering a milestone
        """
        pass

class AveragedModelHook(MilestoneHook):
    """Handles the update of AveragedModel
    First, it uses the swa_utils in Pytorch.
    Second, we follow the approach of Tensorflow wherein the batch norm
    is smoothed out. In contrast to Timm's approach wherein everything is smoothed out.

    This hook is senstive to the position along the Hook. It must be triggered before updating the dataset.
    It will make the calcualation of batch statistics more accurate.
    Args:
        ave_model (torch.optim.swa_utils.AveragedModel): Averaged Model as defined in the training script
            We expect that it is compatible on both Stochastic Weight Averaging (SWA) & Tensorflow's ModelEma
        active_dataset (ActiveDataset): contains active dataset used for active learning
        train_builder (Callable): responsible for building the data loader needed for calculating batch statistics
            Its Sampler must be finite rather than the default Detectron2's TrainingSampler

    Notes:
        - After each milestone, we then update the batch statistics 

    """
    def __init__(self, milestones, ave_model,active_dataset,train_builder):
        super().__init__(milestones)
        self.ave_model = ave_model
        self.active_dataset = active_dataset
        self.train_builder = train_builder

    def after_step(self):
        """Update the model paramters
        """
        model = self.trainer.model
        self.ave_model.update_parameters(model)
        # We trigger first updating of parameters before milestones for consistency
        super().after_step()

    def after_milestone(self):
        data_loader = self.train_builder(self.active_dataset)
        # it loads 
        torch.optim.swa_utils.update_bn(data_loader,self.ave_model)

class AperiodicEvalHook(MilestoneHook):
    """
    Run an evaluation function based on certain checkpoints, and at the end of training.
    It is executed every ``checkpoint`` iterations and after the last iteration.
    """

    def __init__(self, checkpoints, eval_function):
        """
        Args:
            checkpoints (List[int]): the list of checkpoints to run `eval_function`. Set to [] to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        super().__init__(checkpoints)
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)
            plot_list = results.pop("plot", [])
            for cls in plot_list:
                self.trainer.storage.put_image(cls,plot_list[cls])
            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_milestone(self):
        if (self.trainer.iter + 1) != self.trainer.max_iter:
            # do the last eval in after_train
            self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func




class DatasetUpdaterHook(MilestoneHook):
    """Update the dataset after every active step
    Attributes:
        dataset_updater: Dictates on how to update the dataset
        dataset (ActiveDataset): manages all dataset for labeled and unlabed dataset
    Args:
        milestones(List[int]): list of iterations on where this hook will be triggered
    """
    def __init__(self, milestones, updater, active_dataset):
        super().__init__(milestones)
        self.dataset_updater = updater
        self.dataset = active_dataset
        pass

    def after_milestone(self):
        logger = logging.getLogger("activedet.engine.hooks")
        # Note: this update method is a stateful operation
        self.dataset_updater.update(self.dataset)
        logger.info("Finished updating the dataset")


class DropStartHook(MilestoneHook):
    """Drops the initially acquired labeled data.
    Usually, these data are acquired randomly.
    Assumes that you use ActiveDataset
    Attributes:
        dataset (ActiveDataset): manages all dataset for labeled and unlabeled dataset
        checkpointer (Checkpointer): manages state_dict of the whole training regime
        init_path (str): describes on where to save the model weights right after dropping initial
            data. Useful for new active learning cycles/steps
    Args:
        milestones(List[int]): list of iterations on where this hook will be triggered
    """
    def __init__(self, milestones, active_dataset, checkpointer, init_path):
        super().__init__(milestones)
        # Assumes that you are using ActiveDataset
        self.dataset = active_dataset
        self.checkpointer = checkpointer
        self.init_path = init_path
    
    def after_milestone(self):
        logger = logging.getLogger("activedet.engine.hooks")
        if self.trainer.iter + 1 == self._milestones[0]:
            self.dataset.remove_start()
            self.checkpointer.save(self.init_path)
            logger.info("Dropped initial data.Saving trained weights as the intial weights on all iterations.")

class DatasetVisualizerHook(MilestoneHook):
    """Adds Dataset Visualization after each milestone
    It visualizes the labeled dataset
    Assumes that you use ActiveDataset
    Attributes:
        visualizer (DistributionVisualizer): indicates on how to visualize the dataset
        dataset (ActiveDataset): manages all dataset for labeled and unlabeled dataset
    Args:
        milestones(List[int]): list of iterations on where this hook will be triggered
    """
    def __init__(self, milestones, visualizer, active_dataset):
        super().__init__(milestones)
        self.visualizer = visualizer
        # Assumes that you are using ActiveDataset
        self.dataset = active_dataset

    def after_milestone(self):
        logger = logging.getLogger("activedet.engine.hooks")
        # ActiveDataset -> Labelled Dataset
        self.visualizer.add_histogram(self.dataset.dataset)
        logger.info("Successfully added Dataset Histogram on tensorboard")

class DataloaderRebuilderHook(MilestoneHook):
    """Rebuild DataLoader
    After each Acquisition of New Data towards active datasets,
    We need to update the iter(data_loader) iterator and the Sampler
    so that it will be applicable to the evolved dataset.
    Otherwise, the sampler would think that we maintain the same dataset length

    Attributes:
        dataset: Active Dataset used for train, eval, pool
        rebuilder: A callable function that takes
            new dataset as an input and returns a new dataloader
            Example:
                new_data_loader = rebuilder(dataset)
    Args:
        milestones : List on where it would trigger this hook. Typically,
                    It would be a list of iteration where active step is triggered
    """
    def __init__(self, milestones: List[int], dataset: ActiveDataset, dataloader_builder: Callable, kind: str = "labeled"):
        super().__init__(milestones)
        # Assumes that you are using ActiveDataset
        self.dataset = dataset
        self.rebuilder = dataloader_builder
        self.kind = kind
    
    def after_milestone(self):
        logger = logging.getLogger("activedet.engine.hooks")

        data_loader = self.rebuilder(self.dataset)
        # Assumes that it uses ActiveTrainer
        self.trainer.set_dataloader(data_loader,kind=self.kind)
        logger.info("Finished on updating dataloader")


class ModelResetterHook(MilestoneHook):
    """Resets model and optimizer
    This is to follow Yarin Gal's paper

    Attributes:
        checkpointer (Checkpointer): manages state_dict of the whole training regime
        init_path (str): describes on where to save the model weights right after dropping initial
            data. Useful for new active learning cycles/steps
    Args:
        milestones (List[int]) : List on where it would trigger this hook. Typically,
            It would be a list of iteration where active step is triggered      
    """
    def __init__(self, milestones, checkpointer, init_path):
        super().__init__(milestones)
        self.checkpointer = checkpointer
        self.init_path = init_path
    def after_milestone(self):
        logger = logging.getLogger("activedet.engine.hooks")

        self.checkpointer.load(os.path.join(self.checkpointer.save_dir,self.init_path + ".pth"),["model"])
        # For SemiSupervisedTrainer only
        self.trainer.is_reset_optim = True

        model = self.trainer.model
        if isinstance(model, DistributedDataParallel):
        # broadcast loaded data/model from the first rank, because other
        # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                model._sync_params_and_buffers()

        logger.info("Successfully reset into initial weights")

class ValidationLossHook(MilestoneHook):
    """Calculates the validation loss 

    Args:
        data_loader = validation dataloader.
        checkpoints = list of iterations on where it will be triggered. When set as [], 
                    it will run after each step
    Attributes:
        _val_loader = data loader used as the validation dataset
        _checkpoints = list of iteration number where validation loss is calculated
    Notes:
        Dataloader must contain the ground truth similar to training set.
        Validation is performed on a per-iteration basis.
        Thus, the way the loss is interpreted the rate of change
        Its main drawback is when validation set has a mismatch with training set
    """
    def __init__(self, data_loader, checkpoints  = []):
        self._val_loader = data_loader
        self._val_loader_iter = iter(data_loader)
        super().__init__(checkpoints)
    
    def after_milestone(self):
        """Performs validation across checkpoints
        """
        # ExitStack() provides an ability to enter context managers at the same time
        # with the proper order of exit according to Last-in First-out (LIFO) approach
        with ExitStack() as stack:
            # Set model into training mode.
            stack.enter_context(validation_context(self.trainer.model))
            # Disable autograd in order not to affect gradient computation and backpropagation
            stack.enter_context(torch.no_grad())

            data = next(self._val_loader_iter)                
            loss_dict = self.trainer.model(data)

            self._write_metrics(loss_dict)
    
    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        # Perform only on the main process/GPU to avoid duplication
        if comm.is_main_process():
            storage = get_event_storage()

            # average the rest metrics. It's for multi-GPU setup
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            # Calculate the total loss across all loss function
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            # For tensorboard writer
            storage.put_scalar("{}total_loss/validation".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)
