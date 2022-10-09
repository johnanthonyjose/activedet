import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHeadDropout(nn.Sequential):
    """Similar to 'FasterRCNNConvFCHead' from detectron2.
    Modified to contain dropout layers, similar to the implementation of ProbDet (https://github.com/asharakeh/probdet).
    Only the following methods were modified:
        __init__(): Added torch.nn.Dropout module and corresponding input arguments.
        from_config(): Added input arguments to for torch.nn.Dropout module.
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", dropout_rate=0.5,
    ):
        """
        Modified to include torch.nn.Dropout modules when initializing layers.

        Modified to include the following arguments:
            dropout_rate: Probability of element to be zeroed. Used in torch.nn.Dropout module.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            fc_dropout = nn.Dropout(p=dropout_rate)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_dropout{}".format(k + 1), fc_dropout)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        Modified to return the following arguments based on configuration.
            dropout_rate: Probability of element to be zeroed. Used in torch.nn.Dropout module.
        """
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "dropout_rate": cfg.MODEL.ROI_BOX_HEAD.DROPOUT_RATE,
        }

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])