from ..common.optim import SGD as optimizer
from ..common.scheduler.voc_scheduler import lr_multiplier
from ..common.data.pascal_voc import dataloader
from ..common.models.torchvision_retinanet import model
from ..common.train.standard import train


model.backbone.bottom_up.stages.norm = "FrozenBN"
model.backbone.bottom_up.stem.norm = "FrozenBN"
model.backbone.bottom_up.freeze_at = 2
model.num_classes = 20
# linear scaling
dataloader.train.batch_size = 2
optimizer.lr = 0.001
lr_multiplier.scheduler.milestones = [96000,128000]
lr_multiplier.warmup_length = 1600 / 144000
train.max_iter = 144000
train.amp.enabled = True
train.init_checkpoint = "http://download.pytorch.org/models/resnet50-19c8e357.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/scrl_1000ep.pth"
train.output_dir = "output/PascalVOC-Detection/retinanet_torch_R50_FPN_1x/bs2/"
