from typing import Dict, List
import torch.nn as nn
import torch

from detectron2.config import configurable
from detectron2.utils.registry import Registry

AUX_PREDICTOR_REGISTRY = Registry("AUX_PREDICTOR")
AUX_PREDICTOR_REGISTRY.__doc__ = """
Registry for Auxilliary Predictor Modules

Set of Modules that defines Loss Prediction Modules under Learning Loss

"""

@AUX_PREDICTOR_REGISTRY.register()
class LossPrediction(nn.Module):
    """Loss Prediction Module from the paper
    Paper: `Learning Loss for Active Learning` By Yoo and Kweon (2019)
    https://arxiv.org/pdf/1905.03677.pdf
    """

    @configurable
    def __init__(
        self,
        in_features: List[str],
        pooler_resolutions: List[int],
        in_channels_per_feature: List[int],
        out_widths: List[int],
        loss_weight: float,
        margin: int = 1,
    ):
        """
        in_features : names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
        pooler_resolution: The pool resolution that will be utilized to standardize the width/height resolution
        in_channels_per_feature (List[int]): expected channel dimension for each feature
        out_widths: Expected output feature width for each stage
        loss_weight: weighting hyperparameter (the lambda in paper)
        margin: Arbitrary margin. I think it is inspired from Triplet Loss or SVM

        """
        super().__init__()
        self.loss_weight = loss_weight
        self.margin = margin
        self.in_features = in_features
        self.blocks = []
        for in_feat, in_channel, pool_size, out_width in zip(in_features, in_channels_per_feature, pooler_resolutions, out_widths):
            gap = nn.AdaptiveAvgPool2d(pool_size)
            flatten = nn.Flatten()
            linear = nn.Linear(in_channel * (pool_size ** 2), out_width)
            relu = nn.ReLU()
            block = nn.Sequential(gap, flatten, linear, relu)
            self.blocks.append(block)
            self.add_module("loss_pred_{}".format(in_feat), block)

        # Predicting the loss         
        self.loss_predictor = nn.Linear(sum(out_widths),1)

    @classmethod
    def from_config(cls, cfg, backbone_shape):
        ret = {}
        in_features = ret["in_features"] = cfg.MODEL.AUX.LOSS_PREDICTION.IN_FEATURES
        ret["pooler_resolutions"] = cfg.MODEL.AUX.LOSS_PREDICTION.POOLER_RESOLUTIONS
        ret["out_widths"] = cfg.MODEL.AUX.LOSS_PREDICTION.OUTPUT_WIDTH
        ret["margin"] = cfg.MODEL.AUX.LOSS_PREDICTION.MARGIN
        # We include the factor of 2 due to the usage of pairs in a single batch
        ret["loss_weight"] = cfg.MODEL.AUX.LOSS_PREDICTION.LAMBDA
        input_shapes = backbone_shape
        ret["in_channels_per_feature"] = [input_shapes[f].channels for f in in_features]

        return ret


    def forward(self, features, target_loss=None):
        """Forward Propagation
        Args:
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        feat = [features[in_feat].detach() for in_feat in self.in_features]
        # del features
        losses = []
        for f, block in zip(feat, self.blocks):
            block = block.to(f.device)
            loss = block(f)
            losses.append(loss)

        self.loss_predictor = self.loss_predictor.to(feat[0].device)
        # del feat
        pred_loss = self.loss_predictor(torch.cat(losses,dim=1))
        # del losses
        # Squeeze into 1D tensor
        pred_loss = pred_loss.squeeze()

        if self.training:
            actual_loss = self.losses(pred_loss, target_loss, self.loss_weight)
            # del target_loss
            return [], actual_loss
        
        return pred_loss, {}


    def losses(self, pred_loss, target_loss, loss_weight):
        """Calculates the learning loss during training
        Args:
            pred_loss : the predicted loss from loss prediction module with size B
            target_loss: the actual loss from the model training with size B
            loss_weight:  The weighting hyperparameter for the calculated
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_learn"
        """        
        # remove target loss from the computation graph  
        # This is to ensure that succeeding computation doesn't affect the original model being trained
        target = target_loss.detach()
        
        # We start by calculating the pairs
        lossP = (target - target.flip(0))[:target.shape[0] // 2]
        
        one = 2 * torch.sign(torch.clamp(lossP, min=0)) - 1 # 1 operation which is defined by the authors
        
        # We now calculate the predicted loss pairs
        pred_lossP = (pred_loss - pred_loss.flip(0))[:pred_loss.shape[0] // 2]
        
        # The calculation below faithfully follows the equation above
        loss_learn = 2*loss_weight* torch.clamp(-1 * one * pred_lossP + self.margin, min=0)
        return {"loss_learn": loss_learn / (loss_learn.shape[0]*2)} # Normalzed by the original batch size



def build_aux_predictor(cfg, input_shape):
    """
    Build Auxilliary modules as defined by `cfg.MODEL.AUX.PREDICTOR`.
    """

    name = cfg.MODEL.AUX.PREDICTOR
    return AUX_PREDICTOR_REGISTRY.get(name)(cfg, input_shape)