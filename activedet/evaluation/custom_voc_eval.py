# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from signal import pause
import numpy as np
import os, cv2
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

import math

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator, parse_rec, voc_ap

from matplotlib import patches, patheffects
from matplotlib import pyplot as plt

from datetime import datetime 

class CustomPascalVOCDetectionEvaluator(PascalVOCDetectionEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, output_dir =None):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        super().__init__(dataset_name)
        meta = MetadataCatalog.get(dataset_name)
        image_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "JPEGImages/")
        )
        foldername = str(datetime.now()).replace(' ','_')
        plot_out = PathManager.get_local_path(
            os.path.join(output_dir, os.path.join("plot","{}".format(foldername)))
        )
        if not os.path.isdir(os.path.join(output_dir, os.path.join("plot","{}".format(foldername)))):
            os.makedirs(os.path.join(output_dir, os.path.join("plot","{}".format(foldername))))
        self._foldername = os.path.join(output_dir,foldername)
        self._image_file_template = os.path.join(image_dir_local, "{}.jpg")
        self._plot_out = os.path.join(plot_out, "{}.jpg")
        self._output_dir = output_dir

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            recs = defaultdict(list) #
            total_mistakes = defaultdict(list)
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                mistake_list_per_class = defaultdict(list)
                for thresh in range(50, 100, 5):
                    rec, prec, ap, mistake_list = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)
                    mistake_list_per_class["image_name"] = mistake_list_per_class["image_name"] + mistake_list["image_name"]
                    mistake_list_per_class["BB"] = mistake_list_per_class["BB"] + mistake_list["BB"]
                    mistake_list_per_class["BBGT"] = mistake_list_per_class["BBGT"] + mistake_list["BBGT"]
                total_mistakes["{}".format(cls_name)] = mistake_list_per_class
        # get first 10 mistakes per class
        visualization_list = create_mistake_list(total_mistakes,10)
        #create visualizations
        tot_img = create_visualizations(self,visualization_list)
        # create grid

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        ret["plot"] = tot_img
        return ret



def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    mistake_list = defaultdict(list)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
                    #get image, predictedbbox, and gt bbox
                    if image_ids[d]!=[] and BB[d,:]!=[] and BBGT!=[]:
                        mistake_list["image_name"].append(image_ids[d])
                        mistake_list["BB"].append(BB[d,:])
                        mistake_list["BBGT"].append(BBGT)

        else:
            fp[d] = 1.0
            #get image, predictedbbox, and gt bbox
            if image_ids[d]!=[] and BB[d,:]!=[] and BBGT!=[]:
                mistake_list["image_name"].append(image_ids[d])
                mistake_list["BB"].append(BB[d,:])
                mistake_list["BBGT"].append(BBGT)
        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap, mistake_list

def create_mistake_list(mistake_list,length):
    top_mistakes = {}
    for cls in mistake_list:
        class_list = []
        if len(mistake_list[cls]["image_name"]) > length:
            for i in range(length):
                item_dict = {}
                item_dict["image_name"] = mistake_list[cls]["image_name"][i]
                item_dict["BB"] = mistake_list[cls]["BB"][i]
                item_dict["BBGT"] = mistake_list[cls]["BBGT"][i]
                class_list.append(item_dict)
        else:
            for i in range(len(mistake_list[cls]["image_name"])):
                item_dict = {}
                item_dict["image_name"] = mistake_list[cls]["image_name"][i]
                item_dict["BB"] = mistake_list[cls]["BB"][i]
                item_dict["BBGT"] = mistake_list[cls]["BBGT"][i]
                class_list.append(item_dict)
        top_mistakes[cls] = class_list
    return top_mistakes

def fig_to_np(fig,ax):
    #Image from plot
    ax.axis('off')
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot

def create_visualizations(self,image_list):
    tot_img = {}
    for cls in image_list:
        cls_img = []
        for i, item in enumerate(image_list[cls]):
            img = cv2.imread(self._image_file_template.format(item["image_name"]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #create figure
            plt_name = str("{}/{}".format(self._foldername,cls))
            plt_name_save = "{}".format(cls,item["image_name"],i)
            fig, ax = plt.subplots(figsize = (6,9))
            fig.suptitle(item["image_name"], fontsize = 16)
            ax.xaxis.tick_top() 
            ax.imshow(img_rgb)
            x = float(item["BB"][0])
            y = float(item["BB"][1])
            w = float(item["BB"][2])-float(item["BB"][0])
            h = float(item["BB"][3])-float(item["BB"][1])
            for BBGT in item["BBGT"]:
                x_gt = BBGT[0]
                y_gt = BBGT[1]
                w_gt = BBGT[2]-BBGT[0]
                h_gt = BBGT[3]-BBGT[1]
                #add rectange and text for ground truth
                ax.add_patch(patches.Rectangle((x_gt,y_gt),w_gt,h_gt, fill=False, edgecolor='red', lw=1))
                ax.text(x_gt,(y_gt-20),str(cls),verticalalignment='top',color='white',fontsize=10,weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
            #add rectange and text predictions
            ax.add_patch(patches.Rectangle((x,y),w,h, fill=False, edgecolor='green', lw=1))
            ax.text(x,(y-20),str(cls),verticalalignment='top',color='white',fontsize=10,weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
            #plt.savefig(self._plot_out.format(plt_name))
            #load image as numpy array
            np_plot = fig_to_np(fig,ax)
            cls_img.append({"image":np_plot,"name":plt_name})
            plt.close("all")
        if cls_img:
            img_grid = create_grid(cls_img, 4).astype(np.uint8)
            #img_grid = img_grid.transpose((2,0,1))[(2,1,0),:,:]
            tot_img[plt_name] = img_grid.transpose((2,0,1))
            img_grid_bgr = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._plot_out.format(plt_name_save),img_grid_bgr)
    return tot_img


def create_grid(image_list, width):
    f = 0
    for i in range(math.ceil(len(image_list)/width)):
        #create vertical
        for j in range(width):
            if j == 0:
                if f<len(image_list):
                    horizontal_grid = image_list[f]["image"]
                else:
                    horizontal_grid = np.zeros(image_list[0]["image"].shape)
                horizontal_grid = image_list[f]["image"]
            # create horizontal
            else:
                if f<len(image_list):
                    horizontal_grid = np.concatenate((horizontal_grid,image_list[f]["image"]),axis = 1)
                else:
                    horizontal_grid = np.concatenate((horizontal_grid,np.zeros(image_list[0]["image"].shape)),axis= 1)
            f=f+1
        if i == 0:
            img_grid = horizontal_grid
        else:
            img_grid = np.concatenate((img_grid,horizontal_grid),axis = 0)
    return img_grid
