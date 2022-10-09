import math
from typing import Dict, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from fvcore.nn import weight_init
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from torch.nn.modules import activation



@SEM_SEG_HEADS_REGISTRY.register()
class UNetHead(nn.Module):
    """Implements U-Net Head for Semantic Segmentation
    """

    @configurable
    def __init__(self, 
        input_shape: Dict[str, ShapeSpec],
        *, 
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1
        ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.common_stride = common_stride
        _assert_strides_are_log2_contiguous(feature_strides)

        self.upsample_convs = []
        self.fused_convs = []
        # Loop through on how ExpandingBlock needed to be done
        for strides, channels in zip(feature_strides, feature_channels):
            conv_norm = get_norm(norm, channels)
            # In the paper, it's 2x2, but replace with 3x3 for cleaner code
            conv = Conv2d(channels * 2,channels,kernel_size=3, stride=1, padding=1,norm=conv_norm,activation=F.relu)
            weight_init.c2_msra_fill(conv)
            # upsampling = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            #     conv
            # )
            self.upsample_convs.append(conv)

            heads = []
            ins = channels*2
            for k in range(2):
                fusion_norm = get_norm(norm, channels)
                fuse_conv = Conv2d(ins,channels, kernel_size=3, stride=1,padding=1,norm=fusion_norm,activation=F.relu)
                weight_init.c2_msra_fill(fuse_conv)
                heads.append(fuse_conv)
                ins = channels
            fusion = nn.Sequential(*heads)
            self.fused_convs.append(fusion)

            stage = int(math.log2(strides))
            self.add_module("unet_upsample{}".format(stage),conv)
            self.add_module("unet_fusion{}".format(stage),fusion)


        butts = [nn.MaxPool2d(kernel_size=2, stride=2)]
        # Final feature layer. e.g. res5
        ins = feature_channels[-1]
        outs = feature_channels[-1] * 2
        for _ in range(2):
            butt_norm = get_norm(norm, outs)
            conv = Conv2d(ins,outs, kernel_size=3,stride=1,padding=1,norm=butt_norm,activation=F.relu)
            weight_init.c2_msra_fill(conv)
            butts.append(conv)
            ins = outs
        # At the bottom of U
        self.butt_conv = nn.Sequential(*butts)

        # Transform feature map into the right size
        # Each output_channel corresponds to a class label
        self.predictor = Conv2d(conv_dims,num_classes,kernel_size=1, stride=1, padding=0)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
        }
    
    def forward(self, features, targets=None):
        """
        Args:
            features (Dict[str,Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        cur_feature = self.butt_conv(features[self.in_features[-1]])

        # Reverse for easier forward pass
        for upsampling_conv,fused_conv,in_feature in zip(reversed(self.upsample_convs), reversed(self.fused_convs),reversed(self.in_features)):
            upsample = F.interpolate(cur_feature,size=features[in_feature].shape[2:],mode="bilinear",align_corners=True)
            upsample = upsampling_conv(upsample)
            cur_feature = fused_conv(torch.cat([upsample, features[in_feature]], dim=1))

        pred = self.predictor(cur_feature)

        if self.training:
            return None, self.losses(pred, targets)
        else:
            pred = F.interpolate(pred,scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        return pred, {}


    def losses(self, predictions, targets):
        predictions = F.interpolate(predictions,size=targets.shape[-2:], mode="bilinear", align_corners=False) 
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )
