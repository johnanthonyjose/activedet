import re
from detectron2.checkpoint import DetectionCheckpointer

from .scrl_checkpoint import convert_kakao_names
from .torchvision_checkpoint import convert_torchvision_names, convert_torchvision_adobe

class ActivedetCheckpointer(DetectionCheckpointer):
    """Same as `DetectionCheckpointer` but added functionality
    for custom model weights
    """

    def _load_file(self, filename):
        loaded = super()._load_file(filename)

        if filename.endswith(".pth") and re.search("/brainrepo/scrl", filename.lower()):
            loaded["model"] = convert_kakao_names(loaded["model"])
            loaded["matching_heuristics"] = True
            loaded["__author__"] = "KakaoBrain"

        elif filename.endswith(".pth") and (re.search("/models/resnet", filename.lower()) or "torch_" in filename.lower()):
            loaded["model"] = convert_torchvision_names(loaded["model"])
            loaded["matching_heuristics"] = True
            loaded["__author__"] = "torchvision"
        
        elif filename.endswith(".pth") and re.search("resnet50_lpf", filename.lower()):
            loaded["model"] = convert_torchvision_adobe(loaded["model"]["state_dict"])
            loaded["matching_heuristics"] = True
            loaded["__author__"] = "Adobe"

        return loaded