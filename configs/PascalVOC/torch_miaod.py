from .miaod import dataloader, optimizer, lr_multiplier, train, test, active_learning
from ..common.models.torchvision_MIAOD import model


model.backbone.bottom_up.stages.norm = "FrozenBN"
model.backbone.bottom_up.stem.norm = "FrozenBN"
model.backbone.bottom_up.freeze_at = 2
model.num_classes = 20

train.init_checkpoint = "http://download.pytorch.org/models/resnet50-19c8e357.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/scrl_1000ep.pth"
train.output_dir = "output/PascalVOC/torch_miaod/"
train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344