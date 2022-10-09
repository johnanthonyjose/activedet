import numpy as np
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union, Any

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.pascal_voc import CLASS_NAMES


def load_voc_segmentation(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC Segmentation to Detectron2 format.
    Args:
        dirname: Contain "SegmentationClass", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Segmentation", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    segmentation_dirname = PathManager.get_local_path(os.path.join(dirname, "SegmentationClass_detectron2/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        segm_file = os.path.join(segmentation_dirname,fileid + ".png")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
            "sem_seg_file_name": segm_file
        }
        
        dicts.append(r)
    return dicts

def register_pascal_voc_segmentation(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_segmentation(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names, dirname=dirname, year=year, split=split
    )

def register_all_pascal_voc_segmentation(root):
    SPLITS = [
        ("voc_2007_sem_seg_trainval", "VOC2007", "trainval"),
        ("voc_2007_sem_seg_train", "VOC2007", "train"),
        ("voc_2007_sem_seg_val", "VOC2007", "val"),
        ("voc_2007_sem_seg_test", "VOC2007", "test"),
        ("voc_2012_sem_seg_trainval", "VOC2012", "trainval"),
        ("voc_2012_sem_seg_train", "VOC2012", "train"),
        ("voc_2012_sem_seg_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc_segmentation(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"



_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal_voc_segmentation(_root)