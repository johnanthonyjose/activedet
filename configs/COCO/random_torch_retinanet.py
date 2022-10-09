from detectron2.config import LazyCall as L
from activedet.acquisition.heuristics import ClassificationEntropy
from ..common.data.active_coco import dataloader
from ..common.models.torchvision_retinanet import model
from ..common.optim import SGD as optimizer
from ..common.train.active import train
from ..common.acquisition.MIAOD_random import active_learning
from ..common.test.voc import test
from ..common.scheduler.active_voc_standard import lr_multiplier


active_learning.start_n = 117266 // 50
active_learning.ndata_to_label = 117266 // 50
active_learning.pool.max_sample = 117266 // 5
dataloader.train.batch_size = 2
dataloader.pool.batch_size = 2

model.backbone.bottom_up.stages.norm = "FrozenBN"
model.backbone.bottom_up.stem.norm = "FrozenBN"
model.backbone.bottom_up.freeze_at = 2
optimizer.lr = 0.001
optimizer.weight_decay = 5e-4

train.max_iter = 457314
# lr_multiplier.scheduler.steps = [29325, 58625, 87950, 117250, 146575]
lr_multiplier.scheduler.steps = [29325, 89123, 179418, 300186, 451451]
lr_multiplier.warmup_lengths.warmup_iters = 0
train.init_checkpoint = "http://download.pytorch.org/models/resnet50-19c8e357.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/byol_1000ep.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/scrl_1000ep.pth"

train.output_dir = "output/COCO/random_torch_retinanet/"
train.log_period = 50
train.amp.enabled = True
# train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344
# train.seed = 3959168
# train.seed = 2120925

test.is_visualize = False
