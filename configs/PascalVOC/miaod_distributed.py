from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from ..common.models.MIAOD import model
from ..common.optim import SGD as optimizer
from ..common.scheduler.active_voc_standard import lr_multiplier
from ..common.train.active import train
from ..common.data.active_voc import dataloader
from .miaod import active_learning, test

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 20
optimizer.lr = 0.01
optimizer.weight_decay = 0.0005

train.max_iter = 23582
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.output_dir = "output/PascalVOC/miaod_distributed/"
train.min_uncertainty_points = [312, 832, 1820, 2600, 4004, 5044, 6864, 8164, 10394, 11944, 14580, 16390, 19442, 21512]
train.max_uncertainty_points = [416, 936, 1976, 2756, 4212, 5252, 7124, 8424, 10704, 12254, 14942, 16752, 19856, 21926]
train.labeled_points =         [520, 1040, 2132, 2912, 4420, 5460, 7384, 8684, 11014, 12564, 15304, 17114, 20270, 22340]
train.ddp.find_unused_parameters = True
lr_multiplier.scheduler.steps = [1300,3302,5980,9334,13339,18019,23375]
# lr_multiplier.warmup_lengths.warmup_iters = 0