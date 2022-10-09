from detectron2.config import LazyCall as L
from activedet.acquisition.experimental import ConfidenceBinnedEntropy
from .random_torch_retinanet import dataloader,model,optimizer,train,active_learning,test,lr_multiplier

active_learning.heuristic = L(ConfidenceBinnedEntropy)(
    merge="mean",
    threshold=0.8,
    resolution=0.05,
    top_n=1,
    impute_value=1000.0,
)

lr_multiplier.warmup_lengths.warmup_iters = 0
train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/scrl_1000ep.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/byol_1000ep.pth"
train.output_dir = "output/COCO/ConfBinEnt_retinanet/"
# train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344
# train.seed = 3959168
# train.seed = 2120925