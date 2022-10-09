from detectron2.config import LazyCall as L
from ..common.data.active_voc import dataloader
from ..common.models.torchvision_retinanet import model
from ..common.optim import SGD as optimizer
from ..common.train.active import train
from ..common.acquisition.MIAOD_random import active_learning
from ..common.test.voc import test
from ..common.scheduler.active_voc_standard import lr_multiplier

dataloader.train.batch_size = 2
dataloader.pool.batch_size = 2

model.backbone.bottom_up.stages.norm = "FrozenBN"
model.backbone.bottom_up.stem.norm = "FrozenBN"
model.backbone.bottom_up.freeze_at = 2
model.num_classes = 20
optimizer.lr = 0.001
optimizer.weight_decay = 5e-4

train.max_iter = 188058
lr_multiplier.scheduler.steps = [10350,26264,47559,74211,106244,143634,186405]
lr_multiplier.warmup_lengths.warmup_iters = 0
train.init_checkpoint = "http://download.pytorch.org/models/resnet50-19c8e357.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/byol_1000ep.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/scrl_1000ep.pth"

train.output_dir = "output/PascalVOC/random_torch_retinanet/"
train.log_period = 50
train.amp.enabled = True
# train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344
# train.seed = 3959168
# train.seed = 2120925