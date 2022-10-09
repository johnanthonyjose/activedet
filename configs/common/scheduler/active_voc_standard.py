from detectron2.config import LazyCall as L

from activedet.solver.param_scheduler import ActiveMultiStepParamScheduler
from activedet.solver import MultiWarmupParamScheduler, get_multi_warmup_lengths
from activedet.solver.build import get_milestones

lr_multiplier = L(MultiWarmupParamScheduler)(
    scheduler=L(ActiveMultiStepParamScheduler)(
        #Set checkpoitns under lazy trainer
        steps=[756, 1008, 2634, 3134, 5640, 6392, 9768, 10768, 15024, 16276, 21402, 22902, 28908, 30660, 37536, 39536, 47292, 49544, 58170, 60670],
        max_iter="${...train.max_iter}",
        gamma=0.1
    ),
    milestones=L(get_milestones)(
        max_iter="${...train.max_iter}",
        #Set checkpoitns under lazy trainer
    ),
    warmup_lengths=L(get_multi_warmup_lengths)(
        warmup_iters=100, # Small warmup introduced
        max_iter="${...train.max_iter}",
        #Set checkpoints under lazy trainer
    ),
    warmup_factor=0.001,
)
