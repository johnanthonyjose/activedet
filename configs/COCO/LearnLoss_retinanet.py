from detectron2.config import LazyCall as L
from activedet.acquisition.heuristics import YooLearningLoss
from activedet.modeling.meta_arch.model_aux import LearningLoss
from activedet.modeling.meta_arch.aux_predictor import LossPrediction
from .random_torch_retinanet import dataloader,model,optimizer,train,active_learning,test,lr_multiplier

model.reduction = "batchsum"
model = L(LearningLoss)(
    parent_module=model,
    aux=L(LossPrediction)(
        in_features=["p3","p4","p5","p6","p7"],
        pooler_resolutions=[1,1,1,1,1],
        in_channels_per_feature=[256,256,256,256,256],
        out_widths=[128,128,128,128,128],
        loss_weight=1,
        margin=1,
    ),
    in_features=["backbone"], 
    out_features=["loss_cls","loss_box_reg"]
)

active_learning.heuristic = L(YooLearningLoss)(
    impute_value=1000.0,
)

train.init_checkpoint = "http://download.pytorch.org/models/resnet50-19c8e357.pth"
# train.init_checkpoint = "http://arena.kakaocdn.net/brainrepo/scrl/scrl_1000ep.pth"
train.output_dir = "output/COCO/LearnLoss_retinanet/"
# train.seed = 2863257
# train.seed = 55461164
# train.seed = 14509344
# train.seed = 3959168
# train.seed = 2120925