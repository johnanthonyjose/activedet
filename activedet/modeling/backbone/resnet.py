from typing import Type, Callable, Optional, Union
from functools import partial
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from detectron2.layers import CNNBlockBase, get_norm, Conv2d
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone import BasicStem, ResNet

__all__ = ['ResNetv2', 'resnet50']

        
class BasicBlock(CNNBlockBase):
    """BasicBlock from torchvision in detectron2
    """
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super(BasicBlock, self).__init__(in_channels=in_channels,out_channels=planes * self.expansion,stride=stride)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2d(in_channels,planes,kernel_size=3,stride=stride,padding=1,groups=1,bias=False,dilation=1,norm=norm_layer(planes))
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,norm=norm_layer(planes))
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0.0:
            self.dropout1 = nn.Dropout2d(p=self.dropout_rate, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        if self.dropout_rate > 0.0:
            out = self.dropout1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(CNNBlockBase):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super(Bottleneck, self).__init__(in_channels=in_channels,out_channels=planes * self.expansion,stride=stride)
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1= Conv2d(in_channels,width,kernel_size=1,stride=1,bias=False,norm=norm_layer(width))
        self.conv2 = Conv2d(width,width,kernel_size=3,stride=stride,padding=dilation,groups=groups,bias=False,dilation=dilation,norm=norm_layer(width))
        self.conv3 = Conv2d(width,planes * self.expansion, kernel_size=1,stride=1,bias=False,norm=norm_layer(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0.0:
            self.dropout1 = nn.Dropout2d(p=self.dropout_rate, inplace=False)
            self.dropout2 = nn.Dropout2d(p=self.dropout_rate, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        if self.dropout_rate > 0.0:
            out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu(out)

        if self.dropout_rate > 0.0:
            out = self.dropout2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetv2(ResNet):
    """Inherits the MSRA Resnet from Detectron2. However, it adds make_layer functionality in order to add
    the torchvision Resnet. Thereby, this is torchvision Resnet in detectron2 format
    Reference:
        torchvision resnet.py https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    inplanes = 64
    dilation = 1

    @staticmethod   
    def make_layer(block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, norm_layer: nn.Module,
                    stride: int = 1, dilate: bool = False, groups: int = 1, base_width: int = 64, dropout_rate: float = 0.0) -> nn.Sequential:
        downsample = None
        previous_dilation = ResNetv2.dilation
        if dilate:
            ResNetv2.dilation *= stride
            stride = 1
        if stride != 1 or ResNetv2.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(ResNetv2.inplanes, planes * block.expansion, stride=stride,kernel_size=1,bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(ResNetv2.inplanes, planes, stride, downsample, groups,
                            base_width, previous_dilation, norm_layer))
        ResNetv2.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(ResNetv2.inplanes, planes, groups=groups,
                                base_width=base_width, dilation=ResNetv2.dilation,
                                norm_layer=norm_layer, dropout_rate=dropout_rate))

        return layers
    
    @staticmethod
    def make_default_layers(depth, block_class=None, norm="BN", **kwargs):
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
    
        if block_class is None:
            block_class = BasicBlock if depth < 50 else Bottleneck
        planes = [64,128,256,512]
        strides = [1,2,2,2]

        stages = []
        norm_layer = partial(get_norm,norm)
        ResNetv2.inplanes = 64
        ResNetv2.dilation = 1
        for idx, stage_idx in enumerate(range(2,6)):
            stage = ResNetv2.make_layer(block_class,
                        planes[idx],
                        num_blocks_per_stage[idx],
                        norm_layer,
                        stride=strides[idx],
                        **kwargs
                        )
            stages.append(stage)

        return stages


@BACKBONE_REGISTRY.register()
def build_resnet_torchvision_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config using torchvision Bottleneck.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    norm_layer = partial(get_norm,norm)
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    dropout_rate        = cfg.MODEL.RESNETS.DROPOUT_RATE
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        # assert not any(
        #     deform_on_per_stage
        # ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []
    out_channels = 64
    ResNetv2.inplanes = in_channels
    for idx, stage_idx in enumerate(range(2,6)):
        stride = 2 if stage_idx != 2 else 1
        block_class = BasicBlock if depth < 50 else Bottleneck
        stage = ResNetv2.make_layer(block_class,
                                out_channels,
                                num_blocks_per_stage[idx],
                                norm_layer,
                                stride=stride,
                                groups=num_groups,
                                base_width=width_per_group,
                                dropout_rate=dropout_rate,
                                )
        out_channels *= 2
        stages.append(stage)

    return ResNetv2(stem,stages,out_features=out_features,freeze_at=freeze_at)
