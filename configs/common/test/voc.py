from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from activedet.data.visualizer import DistributionVisualizer


test = OmegaConf.create()
test.visualizer = L(DistributionVisualizer)(
    class_names=[
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
)
test.is_visualize = True
test.with_evaluation = True