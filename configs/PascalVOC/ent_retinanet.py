from detectron2.config import LazyCall as L
from activedet.acquisition.heuristics import ClassificationEntropy
from .random_torch_retinanet import dataloader,model,optimizer,train,active_learning,test,lr_multiplier

active_learning.heuristic = L(ClassificationEntropy)(
    merge="mean",
)

train.output_dir = "output/PascalVOC/ent_retinanet/"
# train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344