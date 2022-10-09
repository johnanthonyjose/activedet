from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

lr_multiplier = L(WarmupParamScheduler)(
                scheduler=L(MultiStepParamScheduler)(
                    values=[1.0, 0.1],
                    milestones=[12000, 16000],
                ),
                warmup_length=200 / 18000,
                warmup_method="linear",
                warmup_factor=0.001,
            )