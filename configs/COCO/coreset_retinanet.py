from functools import partial
from detectron2.config import LazyCall as L
from activedet.acquisition.heuristics import ClassificationCoreSet
from activedet.pool import parse_features
from .random_torch_retinanet import dataloader,model,optimizer,train,active_learning,test,lr_multiplier

active_learning.train_out_features = ["res5"] # "res2","res3","res4","res5"

active_learning.pool_evaluator.collate_processor = partial(parse_features, nodes = ["res5"])

active_learning.heuristic = L(ClassificationCoreSet)(
    impute_value=1000.0,
    ndata_to_label = active_learning.ndata_to_label,
)

train.init_checkpoint = "http://download.pytorch.org/models/resnet50-19c8e357.pth"
train.output_dir = "output/COCO/coreset_retinanet/"
# train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344
# train.seed = 3959168
# train.seed = 2120925
