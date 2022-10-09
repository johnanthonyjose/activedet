# Call this to instantiate MetadataCatalog contents
import activedet.data
from .config import add_active_learning_config

# Call these in order to include in Registry from detectron2
from .modeling import UNetHead, ALROIHeads, Res5ROIHeadsTorch, ALRetinaNet, build_resnet_torchvision_backbone, build_resnet_torchvision_fpn_backbone, FastRCNNConvFCHeadDropout
