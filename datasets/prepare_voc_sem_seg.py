#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The main aim is to remove the background class of VOC
To do so, all of the annotation
are shifted to the left by 1, thereby background label of 0 becomes 255.
In PASCAL_VOC, 255 is reservered for unlabelled pixels. background is now merged into 255.
"""
import numpy as np
import os
from pathlib import Path
import tqdm
from PIL import Image


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    img[img == 254] = 255
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    for name in ["VOC2007", "VOC2012"]:
        annotation_dir = dataset_dir / name / "SegmentationClass"
        output_dir = dataset_dir / name / "SegmentationClass_detectron2"
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)