from typing import Mapping, Tuple
import logging
from detectron2.engine import TrainerBase
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from detectron2.utils.events import get_event_storage
from detectron2.engine import AMPTrainer, SimpleTrainer
import detectron2.utils.comm as comm

from .mode import TrainingMode, get_mode


class DataloaderSetterMixin:
    def set_dataloader(self, data_loader: DataLoader, kind="labeled"):
        if kind == "labeled":
            self.data_loader = data_loader
            self._data_loader_iter = iter(data_loader)
        elif kind == "unlabeled":
            self.unlabeled_loader = data_loader
            self._unlabeled_loader_iter = iter(data_loader)    

class ActiveTrainer(AMPTrainer, DataloaderSetterMixin):
    """Adds an ability to update the existing dataloader"""
    pass

class SemiSupervisedTrainer(TrainerBase, DataloaderSetterMixin):
    """Modifies the training step in order to learn on both
    labeled and unlabled dataset. To do so, it modifies
    the `run_step` so that it can re-define the meaning of one iteration
    At the moment, it supports the ff TrainingMode:
        labeled
        MIN_UNCERTAINTY
        MAX_UNCERTAINTY
    """
    def __init__(self, model, data_loader, optimizer, *, unlabeled_loader, amp_enabled=True, optim_resetter=None):
        super().__init__()
        self._mode = None
        self.mode = TrainingMode.LABELED
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.model_mode = TrainingMode.LABELED
        self.is_reset_optim = True
        self._trainer = (AMPTrainer if amp_enabled else SimpleTrainer)(model,data_loader,optimizer)
        self._trainer._write_metrics = self._write_metrics
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.unlabeled_loader = unlabeled_loader
        self._unlabeled_loader_iter = iter(unlabeled_loader)
        self.optim_resetter = optim_resetter

    @property
    def mode(self):
        """At the moment, the only supported mode for this trainer are the ff:
        (1) Labeled
        (2) Max uncertainty
        (3) Min uncertainty
        """
        allowed = (
            TrainingMode.LABELED
            | TrainingMode.MAX_UNCERTAINTY
            | TrainingMode.MIN_UNCERTAINTY
        )
        return self._mode & allowed

    @mode.setter
    def mode(self, m: TrainingMode):
        """Sets the training mode for this model as defined in the paper.
            There are the modes available:
                (1) Labeled Training Set
                (2) Maximize Instance Uncertainty
                (3) Minimize Instance Uncertainty

            The parameters frozen is copied from their repository
        Args:
            m(TrainingMode): Defines the training mode for the model
        """
        logger = logging.getLogger("activedet.engine.train_loop.SemiSupervisedTrainer")
        if (m != self._mode):
            logger.info("Trainer is now set into {} mode.".format(str(m)))
            self.is_reset_optim = True
        self._mode = m
    
    def change_mode_prehook(self):

        mod = self.model
        if isinstance(self.model, DistributedDataParallel):
            mod = self.model.module

        def change_mode(module: nn.Module, input: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
            phase = get_mode(module)
            phase[0] = torch.tensor(self.model_mode,dtype=phase.dtype,device=phase.device)
            # Ensure that the forward propagation and gradient are appropriate 
            # according to the current TrainingMode
            mod.sync_mode_parameters()
            
            if self.is_reset_optim:
                param_groups = self.optim_resetter()
                self.optimizer.param_groups.clear()
                self.optimizer.state.clear()

                # if TrainingMode.LABELED == self.mode:
                #     self.optimizer.defaults['weight_decay'] = 0.0005
                # elif TrainingMode.MIN_UNCERTAINTY in self.mode:
                #     self.optimizer.defaults['weight_decay'] = 0.0005 * 2
                # elif TrainingMode.MAX_UNCERTAINTY in self.mode:
                #     self.optimizer.defaults['weight_decay'] = 0.0005 * 2

                for param_group in param_groups:
                    self.optimizer.add_param_group(param_group)
                self.is_reset_optim = False
            return input

        
        return mod.register_forward_pre_hook(change_mode)
    
    def run_step(self):
        handle_prehook = self.change_mode_prehook()

        if TrainingMode.LABELED in self.mode:
            self.model_mode = TrainingMode.LABELED
            self._trainer._data_loader_iter = self._data_loader_iter
            self._trainer.run_step()

        elif TrainingMode.MAX_UNCERTAINTY in self.mode:
            # Labeled and MAX_UNCERTAINTY
            self.model_mode = TrainingMode.LABELED | TrainingMode.MAX_UNCERTAINTY
            self._trainer._data_loader_iter = self._data_loader_iter
            self._trainer.run_step()
            # Unlabeled and MAX_UNCERTAINTY
            self.model_mode = TrainingMode.UNLABELED | TrainingMode.MAX_UNCERTAINTY
            self._trainer._data_loader_iter = self._unlabeled_loader_iter
            self._trainer.run_step()

        elif TrainingMode.MIN_UNCERTAINTY in self.mode:
            # Labeled and MIN_UNCERTAINTY
            self.model_mode = TrainingMode.LABELED | TrainingMode.MIN_UNCERTAINTY
            self._trainer._data_loader_iter = self._data_loader_iter
            self._trainer.run_step()
            # Unlabeled and MIN_UNCERATINY
            self.model_mode = TrainingMode.UNLABELED | TrainingMode.MIN_UNCERTAINTY
            self._trainer._data_loader_iter = self._unlabeled_loader_iter
            self._trainer.run_step()

        # Post step. Remove hook
        handle_prehook.remove()
    
    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Almost Identical to `SimpleTrainer._write_metrics()`.
        Only modified the last part
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            # Added logs even if it's just a single loss 
            if len(metrics_dict) >= 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        return self._trainer.state_dict()
    
    def load_state_dict(self, state_dict):
        return self._trainer.load_state_dict(state_dict)