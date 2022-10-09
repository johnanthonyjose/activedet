from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.evaluation import COCOEvaluator
from activedet.data import ActiveDataset
from activedet.data.build import build_detection_pool_loader

from .coco import dataloader

dataloader.train.dataset = L(ActiveDataset)(
        dataset=dataloader.train.dataset,
        start_n="${....active_learning.start_n}",
        init_method="Random",
        seed=0,
    )

dataloader.pool = L(build_detection_pool_loader)(
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    batch_size=16,
    num_workers=4,
    sampler="Trivial",
    num_repeats=1
    )

# dataloader.evaluator = L(COCOEvaluator)(
#     dataset_name="${..test.dataset.names}",
#     output_dir="${...train.output_dir}",
# )