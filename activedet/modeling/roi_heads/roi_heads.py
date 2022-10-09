from functools import partial
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.structures import Boxes, Instances
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling import StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    _log_classification_stats,
)
from detectron2.modeling.roi_heads import Res5ROIHeads
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, nonzero_tuple, get_norm
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances

from activedet.layers import cross_entropy
from activedet.modeling.backbone.resnet import Bottleneck, ResNetv2


@ROI_HEADS_REGISTRY.register()
class ALROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        ret["box_predictor"] = FastRCNNOutputLayersAllOut(
            cfg, ret["box_head"].output_shape
        )

        return {
            "box_in_features": ret["box_in_features"],
            "box_pooler": ret["box_pooler"],
            "box_head": ret["box_head"],
            "box_predictor": ret["box_predictor"],
        }

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsTorch(Res5ROIHeads):


    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on
        ResNetv2.inplanes = out_channels // 2
        blocks = ResNetv2.make_layer(
         Bottleneck,
         planes=out_channels // Bottleneck.expansion,
         blocks=3,
         norm_layer=partial(get_norm,norm),
         stride=2,
         groups=num_groups,
         base_width=width_per_group
        )
        return nn.Sequential(*blocks), out_channels



class FastRCNNOutputLayersAllOut(FastRCNNOutputLayers):
    """Similar to `FastRCNNOutputLayers`.
    The difference are the added attributes on the Instances prediction.
    Specifically:
        - softmax prediction (rather than the argmax output alone)
        - proposals

    In addition to that, it also includes the modifcation of prediction loss
    so that it estimates the loss per batch rather than the total loss only.
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        is_reduce: bool = True,
    ):
        """Similar to `FastRCNNOutputLayers`.
        The difference is the addition of reduce argument
        """
        super().__init__(
            input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
        )

        self.is_reduce = is_reduce
        method = {
            True: {"loss_cls": "mean", "loss_box_reg": "sum",},
            False: {"loss_cls": "none", "loss_box_reg": "none",},
        }
        self.reduction_method = method[bool(is_reduce)]

    @classmethod
    def from_config(cls, cfg, input_shape):
        """Added is_reduce configurable argument.
        """
        ret = super().from_config(cfg,input_shape)
        ret["is_reduce"] = cfg.MODEL.ROI_HEADS.IS_REDUCE
        return ret


    def inference(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """ Replaced `fast_rcnn_inference` as defined in this module
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            proposals,
        )

    def losses(self, predictions, proposals):
        """Same as `losses` but added support for returning loss per batch
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0
            )  # Nx4
            assert (
                not proposal_boxes.requires_grad
            ), "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [
                    (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                    for p in proposals
                ],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device
            )
        proposal_lengths = [len(p) for p in proposals]
        losses = {
            "loss_cls": self.box_cls_loss(
                scores, gt_classes, self.reduction_method["loss_cls"], proposal_lengths
            ),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes,
                gt_boxes,
                proposal_deltas,
                gt_classes,
                proposal_lengths,
                reduction=self.reduction_method["loss_box_reg"],
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_cls_loss(self, scores, gt_classes, reduction, proposal_lengths):
        """Calculates the Classification Loss of each boxes for all instances in single batch
        As dicussed in `Fast R-CNN` paper, it calculates the loss in an instance-level

        Added Support of returning batch-style loss rather than scalar value
        Args:
            scores : predicted scores from feedforward in RxK
            gt_classes: target classes as calculated from balanced subsampling of shape R
            reduction (str): reduction method
            proposal_lengths (List[int]): number of proposals for each Image in a batch
                len(proposal_lengths) would be equivalent to SOLVER.IMS_PER_BATCH
                Each item in this list is the number of proposals predicted on that image
        Returns:
            calculated cross_entropy loss.
            if reduction is batch:
                expected returned size is the batch size per device
            otherwise:
                scalar value
        Notes:
            R is a hyperparameter that can be configured in balanced subsampling.
            By default, R is at most 512 per image.
            It defines the number of regions available in an image
        """
        if str(self.is_reduce).lower() == "batch":
            # assert len(scores) % 512 == 0, f"Predicted scores is not a multiple of 512. scores: {len(scores)}. gt: {len(gt_classes)}"
            # assert len(gt_classes) % 512 == 0, f"Ground truth box is not a multiple of 512. gt: {len(gt_classes)}. scores:{len(scores)}"

            loss_cls = cross_entropy(scores, gt_classes, reduction="none")
            batch_loss = []
            it = 0
            for num_proposals in proposal_lengths:
                batch_loss.append(loss_cls[it:it+num_proposals].sum())
                it += num_proposals
            batch_loss = torch.stack(batch_loss,dim=0)
            # It is normalized across number of instances across the batch.
            # This is to follow the convention in regression loss
            return batch_loss / max(gt_classes.numel(), 1.0)
        return cross_entropy(scores, gt_classes, reduction=reduction)

    def box_reg_loss(
        self,
        proposal_boxes,
        gt_boxes,
        pred_deltas,
        gt_classes,
        proposal_lengths,
        reduction="sum",
    ):
        """Same as `box_reg_loss` but added support of returning batch-style loss rather than scalar value
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
            proposal_lengths (List[int]): number of proposals for each Image in a batch
                len(proposal_lengths) would be equivalent to SOLVER.IMS_PER_BATCH
                Each item in this list is the number of proposals predicted on that image
        Returns:
            calculated regression loss (either thru smooth L1 or GIoU)
            if reduction is batch:
                expected returned size is the batch size per device
            otherwise:
                scalar value

        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds], gt_boxes[fg_inds],
            )
            if self.is_reduce == "batch":
                loss_box = torch.zeros_like(proposal_boxes)
                loss_box_reg = smooth_l1_loss(
                    fg_pred_deltas,
                    gt_pred_deltas,
                    self.smooth_l1_beta,
                    reduction="none",
                )
                loss_box[fg_inds] = loss_box_reg
                loss_box_reg = loss_box.sum(dim=1)
                batch_loss = torch.zeros(len(proposal_lengths),device=loss_box_reg.device)
                it = 0
                for i, num_proposals in enumerate(proposal_lengths):
                    batch_loss[i] = loss_box_reg[it:it+num_proposals].sum()
                    it += num_proposals

                loss_box_reg = batch_loss
            else:
                loss_box_reg = smooth_l1_loss(
                    fg_pred_deltas,
                    gt_pred_deltas,
                    self.smooth_l1_beta,
                    reduction=reduction,
                )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            if self.is_reduce == "batch":
                raise NotImplementedError("Copy-paste and test the code above..")
            else:
                loss_box_reg = giou_loss(
                    fg_pred_boxes, gt_boxes[fg_inds], reduction=reduction
                )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    proposals: List[Instances],
):
    """Same as detectron2's fast_rcnn_inference
    Modified by adding proposals argument

    Args:
        same as detectron2's `fast_rcnn_inference` except for proposals
        proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

    Returns:
        same as detectron2's `fast_rcnn_inference`
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            proposals_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape, proposals_per_image in zip(
            scores, boxes, image_shapes, proposals
        )
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    proposals,
):
    """Modified in order to return more attributes in Instances
    
    Args:
        Same as `fast_rcnn_inference_single_image` but added proposals
        proposals (Instances): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

    Returns:
        Same as `fast_rcnn_inference_single_image` but modified attributes of Instances
        Specifically, it added the ff:
            - prob_scores: softmax prediction (rather than the argmax output alone)
            - proposals: region proposals

    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        # proposals = proposals[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # copy scores after removing background class
    scores_copy = scores.clone().detach()

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        # proposals = proposals[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # apply filter to scores_copy, but keeps full score tensor
    scores_copy = scores_copy[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = (
        boxes[keep],
        scores[keep],
        filter_inds[keep],
        # proposals[keep],
    )

    # filter scores_copy based on nms
    scores_copy = scores_copy[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]

    # add scores_copy to result as scores_softmax
    result.prob_scores = scores_copy

    # result.proposals = proposals

    return result, filter_inds[:, 0]
