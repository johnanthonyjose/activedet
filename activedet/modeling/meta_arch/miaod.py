from typing import Dict, List, Tuple
import math
import logging
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from detectron2.structures import Boxes, Instances
from detectron2.layers import batched_nms, cat, nonzero_tuple
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import (
    RetinaNet,
    RetinaNetHead,
    permute_to_N_HWA_K,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.layers import ShapeSpec, get_norm
from detectron2.config import configurable

from activedet.engine import TrainingMode

logger = logging.getLogger(__name__)

class AODHead(nn.Module):
    """
    The DenseHead used in RetinaNet for object classification and box regression in the paper `MI-AOD`
    It has two subnets for the two tasks, with a common structure but separate parameters.
    In addition, it adds f_mil and separates f_cls into f_1 and f_2
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        num_anchors,
        conv_dims: List[int],
        norm="",
        prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__()

        if norm == "BN" or norm == "SyncBN":
            logger.warning(
                "Shared norm does not work well for BN, SyncBN, expect poor results"
            )

        cls1_subnet = []
        cls2_subnet = []
        bbox_subnet = []
        mil_subnet = []
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        for in_channels, out_channels in zip(
            [input_shape[0].channels] + list(conv_dims), conv_dims
        ):
            cls1_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls1_subnet.append(get_norm(norm, out_channels))
            cls1_subnet.append(nn.ReLU())

            cls2_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls2_subnet.append(get_norm(norm, out_channels))
            cls2_subnet.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

            mil_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                mil_subnet.append(get_norm(norm, out_channels))
            mil_subnet.append(nn.ReLU())

        self.cls1_subnet = nn.Sequential(*cls1_subnet)
        self.cls2_subnet = nn.Sequential(*cls2_subnet)

        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.mil_subnet = nn.Sequential(*mil_subnet)

        self.cls1_score = nn.Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.cls2_score = nn.Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        self.bbox_mil = nn.Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls1_subnet,
            self.cls2_subnet,
            self.bbox_subnet,
            self.mil_subnet,
            self.cls1_score,
            self.cls2_score,
            self.bbox_pred,
            self.bbox_mil,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls1_score.bias, bias_value)
        torch.nn.init.constant_(self.cls2_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.RETINANET.NUM_CONVS,
            "prior_prob": cfg.MODEL.RETINANET.PRIOR_PROB,
            "norm": cfg.MODEL.RETINANET.NORM,
            "num_anchors": num_anchors,
        }

    def forward(self, features: List[Tensor], cls_only=False):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits_cls1 = []
        logits_cls2 = []
        bbox_reg = []
        logits_mil = []
        for feature in features:
            logits_cls1.append(self.cls1_score(self.cls1_subnet(feature)))
            logits_cls2.append(self.cls2_score(self.cls2_subnet(feature)))
            logits_mil.append(self.bbox_mil(self.mil_subnet(feature)))
            if cls_only is False:
                bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))

        return logits_cls1, logits_cls2, bbox_reg, logits_mil

def fmil_loss(f_cls, gt_classes, num_classes, is_convert=True):
    """Calculates the loss function for the uncertainty re-weighting branch
    Args:
        f_cls(List[Tensor]): each tensor corresponds the weighted score as calculated by eq. 5
            Each item corresponds to the prediction in a single pyramid level,
            Each Tensor has a shape is (N, Hi * Wi * Ai, K)
        gt_classes (List[Tensor] or Tensor): indicates the classes of each instance of an image
            The total length of the list is N, batch size. Each item consists of ground truth
            classes in tensor, wherein the size of each Tensor, length of ground truth instance in an image.
            When is_convert is False, we expect gt_classess is a LongTensor of shape (N,K)
        num_classes (int): number of classes to be predicted by the model
        is_convert (bool): Whether to convert the gt_classes into multi-label or not. Default: True

    Returns:
        Calculated Multiple Instance Learng Loss, known as l_imgcls in equation (6)
        Expected result is a scalar value as calculated by Binary Cross Entropy
    Reference:
        T. Yuan et al. (2021) Multiple Instance Active Learning for Object Detection. CVPR
    """
    # Expected shape: N x K
    # TODO: Should this be reshaped into regions rather than Batch size?
    valid_masks = [gt >= 0 for gt in gt_classes]

    def convert_to_multilabel(gt, mask):
        """Converts multi-instance classes into a multi-label encoded vector
        Args:
            gt (Tensor): consists of ground truth class (index) in 1D tensor
            mask (Tensor): valid mask
        Returns:
            multi-label encoded vector
        """
        gt_onehot = F.one_hot(gt[mask], num_classes + 1)[:, :-1] # no loss for the last (background) class
        return gt_onehot.max(dim=0)[0]
    
    if is_convert:
        gt_labels_target = [convert_to_multilabel(gt,mask) for gt, mask in zip(gt_classes, valid_masks)]
        gt_labels_target = torch.stack(gt_labels_target, dim=0) 
    else:
        gt_labels_target = gt_classes

    # Sum across all weighted pred f_cls for each region.
    # Expected shape: List[Tensor]. each Tensor is N x K. Total items: F, num of pyramids
    # f_cls = [level.sum(dim=1) for level in f_cls]

    # Expected shape: N x F x K
    # f_cls = torch.stack(f_cls,dim=1)


    # Take maximum estimate across all levels. Specifically,
    # given the sum of estimates of each region, we only value those who can provide maximum estimate
    # The assumption is that the highest class score should provide better estimate of the class prediction
    # If it is not true, then it calculates a gradient on that highest score to fine-tune its prediction
    # If in the next iteration, it is not the highest gradient, then it means it is not the right feature.
    # We then expect the better candidate to be the highest one to estimate the ground truth
    # If in this stage, it became the best candidate, then we expect it to have low gradient but continually
    # be the candidatee for the image-label pair combination
    # TODO: Should this be converted into logits similar to focal loss?
    # Answer: Yes. But we want to wait and reproduce faithfully first for bug fixing
    # Update: I have to use sigmoid instead in order to make it compatible with mixed precision
    fmax = torch.zeros(gt_labels_target.shape[0],num_classes).cuda(torch.cuda.current_device())
    for f_single in f_cls:
        # Each level would be part of the computation graph rather than vectorized version
        fmax = torch.max(fmax,f_single.sum(dim=1))
    fmax = fmax.clamp(1e-5, 1.0-1e-5)
    fmax_logits = torch.log(fmax / (1 - fmax))
    if fmax_logits.isnan().sum() > 0:
        num_nans = fmax_logits.isnan().sum()
        total = fmax_logits.numel()
        raise ValueError("It should not have nan. It has {}/{}".format(num_nans,total))

    # TODO: Should we add a mask?
    # TODO: Explore the usage of focal loss for this context
    loss = F.binary_cross_entropy_with_logits(fmax_logits, gt_labels_target.float())
    return loss


@META_ARCH_REGISTRY.register()
class MIAOD(RetinaNet):
    """
    Implements the modifieda RetinaNet in :paper:`MI-AOD`.
    Attributes:
        (All attributes of RetinaNet)
        reg_lambda (float): lambda used as regularization hyparameter as seen in their paper
        phase (torch.Tensor): Indicates the training phase of the model
        mode (TrainingMode): The training phase in read-only Enum format
    
    """

    def __init__(
        self,
        *,
        backbone,
        head: nn.Module,
        head_in_features,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
        reg_lambda=0.5,
    ):
        super().__init__(
            backbone=backbone,
            head=head,
            head_in_features=head_in_features,
            anchor_generator=anchor_generator,
            box2box_transform=box2box_transform,
            anchor_matcher=anchor_matcher,
            num_classes=num_classes,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            test_score_thresh=test_score_thresh,
            test_topk_candidates=test_topk_candidates,
            test_nms_thresh=test_nms_thresh,
            max_detections_per_image=max_detections_per_image,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            vis_period=vis_period,
            input_format=input_format,
        )
        self.reg_lambda = reg_lambda
        # Sets training mode
        # It is registered as buffer in order to let DDP handle synchronization across devices
        # The trick is have a tensor of size 1 so that you can perform in-place operation
        # The method to change the TrainingPhase is to do the ff:
        #   phasor = get_mode(model)
        #   phasor[0] = torch.tensor(TrainingMode.YourMode,device=phasor.device)
        # In this way, it changes the Phase operation
        self.register_buffer("phase",torch.tensor([TrainingMode.LABELED], dtype=torch.int64,device=self.device))

        # Put back the old detectron2 behavior wherein it's unstable
        # self.loss_normalizer_momentum = 0.0

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = AODHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransform(
                weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS
            ),
            "anchor_matcher": Matcher(
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                cfg.MODEL.RETINANET.IOU_LABELS,
                allow_low_quality_matches=True,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "head_in_features": cfg.MODEL.RETINANET.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.RETINANET.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
            # TODO: Add lambda discrepancy
        }

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        cls_only = bool(self.mode & TrainingMode.UNLABELED)
        anchors = self.anchor_generator(features)
        (
            pred_logits_cls1,
            pred_logits_cls2,
            pred_anchor_deltas,
            pred_logits_mil,
        ) = self.head(features,cls_only=cls_only)
        y12_ave = [
            (cls1.detach() + cls2.detach()) / 2
            for cls1, cls2 in zip(pred_logits_cls1, pred_logits_cls2)
        ]

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits_cls1 = [
            permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_cls1
        ]
        pred_logits_cls2 = [
            permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_cls2
        ]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        # reshape to permute along class
        # (B,AxK,H,W) -> (B,A,K,H,W) -> (B, H, W, A, K) -> (B, HxWxA, K)
        # This is their weird reshape
        # y_fmil = [x.permute(0,2,3,1).reshape(pred_logits_mil[0].shape[0], -1, self.num_classes) for x in pred_logits_mil]
        # y12_ave = [x.permute(0,2,3,1).reshape(y12_ave[0].shape[0], -1, self.num_classes) for x in y12_ave]
        y_fmil = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_mil]
        y12_ave = [permute_to_N_HWA_K(x, self.num_classes) for x in y12_ave]

        # Equation (5)
        # Q1:Why take the max sigmoid rather than softmax?
        # Ans: I think the main reason is to frame its classification as class-agnostic
        # Then, we focus on the maximal value along the class rather than the score relative to others
        # We don't care on which class it should belong to. Rather, we are only interested on
        # the maximum likelihood estimate it can provide. So, we treat class probabilities independently
        # rather than the conventional categorical distribution
        # Q2: Why take softmax along the instances?
        # A: The trick is that you want to maximize your score when you have better confidence relative to other
        # instances predicted in the field. Therefore, you are trying to amplify the relative scores of a certain
        # instance prediction score because of the softmax.
        # That is, this softmax is to amplify those boxes that have better confidence relative to others
        y_cls = [
            f_mil.softmax(dim=2) * y12.sigmoid().max(2, keepdim=True)[0].softmax(dim=1)
            for f_mil, y12 in zip(y_fmil, y12_ave)
        ]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            # Regardless of the training phase, as long as it's a labeled set, we do this
            # Otherwise, we expect that it is an unlabeled dataset
            if TrainingMode.LABELED in self.mode:
                assert (
                    "instances" in batched_inputs[0]
                ), "Instance annotations are missing in training!"
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

                gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

                losses1 = self.losses(
                    anchors, pred_logits_cls1, gt_labels, pred_anchor_deltas, gt_boxes
                )
                # TODO: Do we add loss normalizer?
                gt_classes = [instance.gt_classes for instance in gt_instances]
                losses1["loss_img_cls"] = fmil_loss(y_cls, gt_classes, self.num_classes) # / self.loss_normalizer
                losses2 = self.losses(
                    anchors, pred_logits_cls2, gt_labels, pred_anchor_deltas, gt_boxes
                )
                # TODO: Do we add loss normalizer?
                losses2["loss_img_cls"] = fmil_loss(y_cls, gt_classes, self.num_classes) # / self.loss_normalizer
                losses = {
                    key: (losses1[key] + losses2[key]) / 2 for key in losses1.keys()
                }
            # Since it's unlabeled dataset, if MAX_UNCERTAINTY is there, we do this
            elif TrainingMode.MAX_UNCERTAINTY in self.mode:
                # TODO: Merge the two training modes. The reason is due to the introduction
                # of training modes. Separate the function is already redundant
                losses = self.loss_discrepancy_max(
                    pred_logits_cls1, pred_logits_cls2, y_cls
                )
            # Since it's unlabeled dataset, if MIN_UNCERTAINTY is there, we do this
            elif TrainingMode.MIN_UNCERTAINTY in self.mode:
                losses = self.loss_discrepancy_min(
                    pred_logits_cls1, pred_logits_cls2, y_cls
                )
                pseudo_label = self.predict_pseudo_labels(pred_logits_cls1,pred_logits_cls2)
                # When at least one of the images in the batch doesn't have at least one 0.5 confidence prediction,
                # We don't train on pseudo labelling at all.
                if (pseudo_label.sum(dim=1) == 0).sum() > 0:
                    pass
                else:
                    losses["loss_img_cls"] = fmil_loss(y_cls,pseudo_label,self.num_classes, is_convert=False)


            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, y12_ave, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            processed_results = []
            if TrainingMode.INFER_UNCERTAINTY in self.mode:
                #Expected output shape: List[Tensor] where each Tensor has (N, Hi*Wi*A,K)

                #TODO: Combine this with NMS aka regular post-processing
                # The idea is that we want to ensure on how it can match across other boxes
                # This can filtered out by performing post-processing step such as NMS
                classifier1 = cat(pred_logits_cls1,dim=1).sigmoid()
                classifier2 = cat(pred_logits_cls2,dim=1).sigmoid()
                mil_weights = cat(y_cls,dim=1)
                # uncertainty = self.discrepancy(pred_logits_cls1,pred_logits_cls2,"min")
                # uncertainty = torch.cat(uncertainty,dim=1)
                
                for cls1,cls2,mil, input_per_image, image_size in zip(
                    classifier1,classifier2, mil_weights, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    instance = Instances((height,width))
                    instance.proba_scores1 = cls1
                    instance.proba_scores2 = cls2
                    instance.mil_weight = mil
                    processed_results.append({"instances": instance})
            # By default, inference will perform the default bbox prediction                
            else:
                results = self.inference(
                    anchors, pred_logits_cls1, pred_anchor_deltas, images.image_sizes
                )
                if torch.jit.is_scripting():
                    return results
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
            return processed_results

    @property
    def mode(self):
        """The only supported mode for this RetinaNet are the ff:
        (1) Labeled
        (2) Max uncertainty
        (3) Min uncertainty
        """
        return self.phase.item() & (
            TrainingMode.LABELED
            | TrainingMode.UNLABELED
            | TrainingMode.MAX_UNCERTAINTY
            | TrainingMode.MIN_UNCERTAINTY
            | TrainingMode.INFER_UNCERTAINTY
            | TrainingMode.INFER_BBOX
        )

    def predict_pseudo_labels(self, y_f1, y_f2):
        """Calculates the pseudo label according to equation 10.
        It is when the trainingmode is unlabeled
        Args:
            y_f1 (List[Tensor]): predicted logits of classifier 1.
                Each tensor represents one pyramid level, with shape (N,Hi*Wi*A,K)
            y_f2 (List[Tensor]): predicted logits of classifier 2
                Each tensor represents one pyramid level, with shape (N,Hi*Wi*A,K)
        Returns:
            pseudo_labels in 1D tensor format
        """
        pseudo_pyramids = [(y1.detach().sigmoid()+y2.detach().sigmoid()) for y1,y2 in zip(y_f1, y_f2)]
        pseudo_proba = torch.cat(pseudo_pyramids,dim=1).max(dim=1)[0] / 2
        pseudo_proba = pseudo_proba.clamp(min=0)
        pos= pseudo_proba >=0.5
        # neg= pseudo_proba < 0.5
        # pseudo_proba[pos] = 1
        # pseudo_proba[neg] = 0
        return pos.long()

    def sync_mode_parameters(self):
        """Synchronize the mode parameters according to the appropriate TrainingMode
            These are the modes available:
                (1) Labeled Training Set
                (2) Maximize Instance Uncertainty
                (3) Minimize Instance Uncertainty

            The parameters frozen is copied from their repository
        Args:
            mode(TrainingMode): Defines the training mode for the model
        """
        # Make this work only when it is solely a labeled set training
        if TrainingMode.LABELED == self.mode:
            for p in self.parameters():
                p.requires_grad = True
        # As long as there's a MAX_UNCERTAINTY flag, make this work
        elif TrainingMode.MAX_UNCERTAINTY in self.mode:
            for name, child in self.named_modules():
                if "head.cls" in name:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False
        # As long as there's a MIN_UNCERTAINTY flag, make this work
        elif TrainingMode.MIN_UNCERTAINTY in self.mode:
            for name, child in self.named_modules():
                if "head.cls" in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True

    def discrepancy(self, y_f1, y_f2, mode: str) -> List[torch.Tensor]:
        """Calculates the discrepancy according to eq. (3)
        It does not contain any re-weighting.
        Returns:
            A list of tensors wherein each item indicates discprancy on that feature level
            Each Tensor has a shape of (N, Hi*Wi*A, K)
        """
        # TODO: WHy is it that it doesn't contain re-weighting?
        y_f1 = [y1.sigmoid() for y1 in y_f1]
        y_f2 = [y2.sigmoid() for y2 in y_f2]

        if mode == "min":
            dis = [torch.abs(y1 - y2) for y1,y2 in zip(y_f1, y_f2)]
        elif mode == "max":
            dis = [1 - torch.abs(y1 - y2) for y1,y2 in zip(y_f1, y_f2)]

        return dis

    def discrepancy_loss(self, y_f1, y_f2, y_cls, mode: str) -> torch.Tensor:
        """Calculates the discrepancy loss, l_dis according to eq. (7)
        Args:
            y_f1 (List[Tensor]): predicted logits of classifier 1.
            y_f2 (List[Tensor]): predicted logits of classifier 2
            y_cls (List[Tensor]): score vector for re-weighting the importance of the discrepancy
            mode (str): Mode of the loss if instance uncertainty learning is meant for either `max` or `min`
        Returns:
            calculated discprenacy loss
        Notes:
            Each element in the list corresponds to one level and has shape (N, Hi * Wi * Ai, K).
            Where K is the number of classes used in `pred_logits_cls1`.

            It supports two modes, maximizing for eq. (8) and minimizing for eq. (9).
        """
        l_dis = self.discrepancy(y_f1,y_f2,mode)
        l_dis = cat([l.view(-1, self.num_classes) for l in l_dis], dim=0)
        w = cat([y.view(-1, self.num_classes).detach() for y in y_cls],dim=0)


        # TODO: Convert into batch-style for possibly learning loss
        l_dis = (l_dis * w).mean(dim=1).sum() * self.reg_lambda
        return l_dis

    def loss_discrepancy_min(self, pred_logits_cls1, pred_logits_cls2, y_cls):
        """Loss function for minimizing discrepancy as defined in eq. (9)
        This is primarily expected to be used when minimizing instance uncertainty on unlabeled dataset
        Args:
            pred_logits_cls1 (List[Tensor]): predicted logits of classifier 1
            pred_logits_cls1 (List[Tensor]): predicted logits of classifier 2
            y_cls(List[Tensor]): predicted score vector for re-weighting the importance of the discrepancy
        Notes:
            Each element in the list corresponds to one level and has shape (N, Hi * Wi * Ai, K).
                Where K is the number of classes used in `pred_logits`.
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_wave_min"
        """
        l_dis = self.discrepancy_loss(pred_logits_cls1, pred_logits_cls2, y_cls, "min")
        return {"loss_disc_min": l_dis}

    def loss_discrepancy_max(self, pred_logits_cls1, pred_logits_cls2, y_cls):
        """Calculates L_wave_max as seen in eq. (8)
        This is primarily expected to be used when maximizing instance uncertainty on unlabeled dataset
        Args:
            pred_logits_cls1, pred_logits_cls2: both are List[Tensor]. predicted logits on adversarial
                classifiers
            y_cls (List[Tensor]): each tensor corresponds the weighted score as calculated by eq. 5
                Each item corresponds to the prediction in a single pyramid level,
        Notes:
            Each element in the list corresponds to one level and has shape (N, Hi * Wi * Ai, K).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_wave_max"
        """
        l_dis = self.discrepancy_loss(pred_logits_cls1, pred_logits_cls2, y_cls, "max")
        return {"loss_disc_max": l_dis}
