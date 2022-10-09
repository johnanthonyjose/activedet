import torch
import torch.nn as nn
from detectron2.utils.registry import Registry
from detectron2.modeling import build_model as d2_build_model
from detectron2.config import CfgNode, configurable

from activedet.utils import LayerHook
from .aux_predictor import build_aux_predictor


META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip


META_ARCH_AUX_REGISTRY = Registry("META_ARCH_AUX")
META_ARCH_AUX_REGISTRY.__doc__ = """
Registry for Meta Architecture with Auxilliary Modules
It defines how each auxillary modules are combined with original model

For instance, it can be combined similar to LearningLoss
when you would want the output of the original model to be the target
of the auxilliary model

"""
@META_ARCH_AUX_REGISTRY.register()
class LearningLoss(nn.Module):
    """Defines Any Model that would include
    a Learning Loss module

    Attributes:
        parent_module (nn.Module): describes the original model to be trained
        aux_lloss (nn.Module): Describes the auxillary module as a learning loss module
        layer_hook (LayerHook): responsible for collecting intermediate layer outputs
    """
    @configurable
    def __init__(self, parent_module, aux,in_features=["backbone"],out_features=["loss_cls","loss_box_reg"]):
        """
        parent_module (nn.Module): describes the original model to be trained
        aux (nn.Module): Describes the auxillary module as a learning loss module
        """
        super().__init__()
        self.parent_module = parent_module
        self.aux_lloss = aux
        self.layer_hook = LayerHook()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_hook.register(parent_module,in_features)
        # detacher is added via hook
        self.detacher = None

        assert in_features, "in_features cannot be empty"
        assert out_features, "out_features cannot be empty"
    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["parent_module"] = model = d2_build_model(cfg)
        ret["aux"] = build_aux_predictor(cfg, model.backbone.output_shape())
        ret["out_features"] = cfg.MODEL.AUX.OUT_FEATURES
        ret["in_features"] = cfg.MODEL.AUX.IN_FEATURES
        return ret

    def forward(self, x):
        """
        """
        if self.training:
            # In this case, we assume that the model was built so that the 
            # ouput loss is predicted per image in a batch
            loss = self.parent_module(x)
            features = self.layer_hook.outputs[0]
            self.layer_hook.clear()
            if self.detacher:
                features = self.detacher(features)

            # We only take the Classification and Regression Loss at the ROI Head.
            # In case there's RPN, We would assume that the RPN loss is not included in the learning loss
            target_loss = torch.stack([loss[out] for out in self.out_features],dim=0).sum(dim=0)
            _, lloss = self.aux_lloss(features, target_loss)
            
            # Thus, at the end we'll manually perform the accumulation so
            # that each loss becomes a scalar value similar to the original implementation
            # All B-sized losses from the model is normalized across all total instances in a batch.
            # Thus, we can simply sum them accordingly.
            loss.update(lloss)
            loss = {key: loss_batch.sum() for key, loss_batch in loss.items()}
            return loss
        
        pred = self.parent_module(x)
        features = self.layer_hook.outputs[0]
        pred_loss, _ = self.aux_lloss(features)
        self.layer_hook.clear()

        # Hack in order to fit into the Instances Data structure
        # The current learning loss is image-level rather than instance-level
        if len(pred_loss.shape) == 0:
            pred_loss = pred_loss.unsqueeze(0)
        for predictions, lloss in zip(pred, pred_loss):
            predictions["instances"].pred_loss = lloss.repeat(len(predictions["instances"]))

        return pred



def build_meta_arch_aux(cfg: CfgNode) -> nn.Module:
    """
    Auxilliary Modules is called thru "MODEL.AUX.NAME"
    By default, when it's none, there won't be any auxilliary modules
    Args:
        cfg (CfgNode) : All-in-one detectron2 config file
    Returns:
        Returns model combined with the auxilliary module
    """
    aux = cfg.MODEL.AUX.NAME
    model = META_ARCH_AUX_REGISTRY.get(aux)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

