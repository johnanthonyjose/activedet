from functools import partial
from detectron2.config import LazyCall as L

from activedet.pool import infer_uncertainty_context, parse_instance
from activedet.acquisition.heuristics import Discrepancy
from ..common.models.MIAOD import model
from ..common.optim import SGD as optimizer
from ..common.acquisition.YooKwon_random import active_learning
from ..common.scheduler.active_voc_standard import lr_multiplier
from ..common.train.active import train
from ..common.data.active_voc import dataloader
from ..common.test.voc import test

dataloader.train.batch_size = 2
dataloader.pool.batch_size = 2
model.backbone.bottom_up.stages.norm = "FrozenBN"
model.backbone.bottom_up.stem.norm = "FrozenBN"
model.backbone.bottom_up.freeze_at = 2
model.num_classes = 20
optimizer.lr = 0.001
optimizer.weight_decay = 5e-4

active_learning.start_n = 16551 // 20
active_learning.ndata_to_label = 16551 // 40
active_learning.epoch_per_step = 26
active_learning.pool.max_sample = -1
active_learning.start_pool_transform = None
active_learning.pool_evaluator.pool_context = infer_uncertainty_context
active_learning.pool_evaluator.collate_processor = partial(parse_instance, device="cuda")
# Fixed into L2 pow as acquisition
active_learning.heuristic = L(Discrepancy)(
    merge="mean",
    threshold=0.0,
    top_n=10000,
)

train.max_iter = 188058
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.output_dir = "output/PascalVOC/miaod/"
train.min_uncertainty_points = [2484, 6624, 14484, 20684, 31846, 40116, 54584, 64914, 82684, 95084, 116160, 130620, 154998, 171528]
train.max_uncertainty_points = [3312, 7452, 15724, 21924, 33500, 41770, 56650, 66980, 85164, 97564, 119052, 133512, 158304, 174834]
train.labeled_points = [4140,8280,16964,23164,35154,43424,58716,69046,87644,100044,121944,136404,161610,178140]
train.log_period = 50
train.amp.enabled = False
lr_multiplier.scheduler.steps = [10350,26264,47559,74211,106244,143634,186405]
lr_multiplier.warmup_lengths.warmup_iters = 0
train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344