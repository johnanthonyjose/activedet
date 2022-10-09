import itertools
import random
from typing import Dict, List, Generator, Union
from abc import ABC, abstractmethod
import logging
from detectron2.config.config import configurable
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import pairwise_iou
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
import detectron2.utils.comm as comm

from .enum import LabelCenterMode
from .distance import batched_all_pairs_l2_dist, dist_score_function


def trivial_merge(x, *args, **kwargs):
    return x


merge_mode = {
    "max": torch.max,
    "min": torch.min,
    "mean": torch.mean,
    "sum": torch.sum,
    "none": trivial_merge,
}


def entropy(p, dim=-1):
    """Computes entropy of input tensor p, which is assumed to be a list of probability distributions.
    Args:
        p: Input tensor containing the probability distribution of multiple instances/detections.
           Has size NxC, where: N = number of instances/detections. K = number of classes.
    Returns:
        entropy: Torch 1D tensor containing the entropy of each instance in the input tensor p.
    """
    assert len(p.shape) == 2, "tensor must be in NxK format"
    p_norm = p / p.sum(dim=dim).unsqueeze(dim=dim)
    return -1 * torch.sum(torch.xlogy(p_norm, p_norm), dim=dim)


def compute_entropy(p, reduction="mean"):
    """Computes the entropy
    Let:
        B = batch size
        N = number of samples
        K = number of classes
        H = height
        W = width
    Args:
        p(torch.Tensor) = BxNxKxHxW prediction map

    Returns:
        calculated scores in BxHxW format (torch.Tensor)
    """
    assert len(p.shape) == 5, "tensor must be in BxNxKxHxW format"
    # Reducers <= reduction of class map
    reducers = {
        "mean": lambda x: torch.mean(x, dim=1),
        "min": lambda x: torch.min(x, dim=1),
        "max": lambda x: torch.max(x, dim=1),
    }
    reducer = reducers[reduction]

    return reducer(-1 * torch.sum(p * torch.log2(p), dim=1))  # remove MC dimension


def calculate_localization_tightness(boxes, proposal_boxes):
    """Calculates the Localization Tightness score of an image based on 'box' and 'proposal box'

    Args:
        boxes: Boxes N x 4
        proposal_boxes: Boxes N x 4

    Returns:
        T: torch.Tensor N
    """
    T = torch.diag(pairwise_iou(boxes, proposal_boxes))
    return T


class Heuristic(ABC):
    def __init__(self, cfg):
        pass

    @abstractmethod
    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ) -> torch.Tensor:
        raise NotImplementedError("Heuristic is meant to be used as abstract")


class SemSegHeuristic:
    def __init__(self, cfg):
        self.cls_heuristic = None
        grid_size = cfg.ACTIVE_LEARNING.POOL.GRID_SIZE
        self.reducer = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        assert (
            comm.get_world_size() == 1
        ), "SemSegHeuristic is only tested with Single GPU"

    def __call__(
        self, proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None]
    ):
        """Invokes the heuristic which ranks the pool images
        Args:
            proba_mapper (Generator[Dict[str, List[torch.Tensor]]]) = probability mapper in lazy loading format
        Returns:
            ranks of each images
        Notes:
            The usage of generator provides a lazy format so that the data does not overwhelm the CPU RAM.
            It requires the usage of iter() and next().
            Alteratively, just feed it in a for loop
        Examples:
            ```python
            for preds in proba_map:
                for name, pred_maps in preds.items():
                    print("image ID: ",name)
                    print("Amount of Monte Carlo Samples:", len(pred_maps))
            ```
            - preds is a dictionary.
                - key(str) : "image_id"
                - value(List[torch.Tensor]): Samples of Monte Carlo Prediction
            - for each iteration of the outer loop, pytorch starts to infer the prediction as defined in evaluation/mc_pool.py
        """
        scores = {}
        for proba_map in proba_mapper:
            # TODO: Convert into functional form
            # e.g. stack dictionary tensor as batch size dimension
            for idx, (name, MCproba_map) in enumerate(proba_map.items()):
                # Assumes all maps is of the same size
                # Softmax converts the raw logits into probalistic map
                # MCsize x classmap x H x W
                MC_proba_map = F.softmax(torch.stack(MCproba_map, dim=0), dim=1)
                # Reduce HxW size into grid_sizexgrid_size
                MC_proba_map = self.reducer(MC_proba_map)
                uncertainty_map = compute_entropy(
                    MC_proba_map[None, ...], reduction="mean"
                )

                # Get mean uncertainty across regions
                score = uncertainty_map.view(uncertainty_map.shape[0], -1).mean(dim=-1)
                scores[name] = {"score": score, "id": idx}
        # Sort by the highest uncertainty
        return torch.tensor(
            [
                v["id"]
                for k, v in sorted(scores.items(), key=lambda item: -item[1]["score"])
            ],
            dtype=torch.long,
        )


class RandomHeuristic:
    """Randomly selects which data points to select
    It is now compatible for distributed training
    Attributes:
        generator (torch.Generator): seed function used to control deterministism in randomization across all workers
    """

    @configurable
    def __init__(self, seed=None):

        if seed is None:
            seed = comm.shared_random_seed()

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        return ret

    def __call__(
        self, proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None]
    ):
        score_length = []
        # Iterate across each batch
        for maps in proba_mapper:
            # Gather all predictions from all GPUs. Then Flatten it as one list
            #TODO: Learn which caused the error below
            assert isinstance(maps, dict), "PoolEvaluator must a dict object. Not {}".format(maps)
            preds = list(itertools.chain.from_iterable(comm.all_gather(maps)))
            # Add the count to the list. The count is dependent on the total batch size (as stated in pool batch size)
            # It could also be lower than pool batch size especially at the last batch
            score_length.append(len(preds))
        # For all batches take the total size of all batches.
        score_length = sum(score_length)
        # Create a tensor from 0 ... n. It's dependent on cfg.ACTIVE_LEARNING.N_DATA_TO_LABEL
        ranks = torch.arange(score_length)
        # Randomize the order of their ranks
        idx = torch.randperm(ranks.nelement(), generator=self.generator)
        ranks = ranks[idx]

        return ranks


class ClassificationEntropy(Heuristic):
    """
    Invokes heuristic which ranks the pool of images based on the entropy of each detection.

    Steps:
        1. Compute entropy for probability distribution (softmax scores) of each detection.
        2. Merge entropy scores of each detection to obtain one score for an MC run.
        3. Merge entropy scores of each MC run to obtain one score for an image.
        4. Rank scores of each image from highest to lowest, highest being the most uncertain.

    Args:
        proba_mapper: Generator yielding batches of images, where each image contains results from multiple MC Dropout runs.
                      This acts as the pool dataset for active learning.

    cfg parameters:
        self.merge_scores: {max, min, mean, sum} How scores from multiple detections are merged. Also used for merging scores from multiple MC Dropout runs.
        self.scores_thresh: Detections with confidence scores lower than this threshold are ignored.
        self.top_n: N number of detections with the highest entropy in an image to merge.

    Returns:
        rank: A 1D tensor containing indices (with respect to the pool) arranged from highest to lowest based on the computed uncertainty score.
    """

    @configurable
    def __init__(self, *, merge="none", threshold=0, top_n=0, impute_value=1000):
        self.merge_scores = merge_mode[merge.lower()]
        self.scores_thresh = threshold
        self.top_n = top_n
        # Used for instances that has no predictions
        # We don't discard because it ruins the order
        # We replace no predictions with the impute_value
        self.impute_value = torch.tensor([impute_value], dtype=torch.float)
        assert (
            comm.get_world_size() == 1
        ), "ClassificationEntropy is only tested with Single GPU"

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["merge"] = cfg.ACTIVE_LEARNING.CLS_MERGE_MODE
        ret["threshold"] = cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH
        ret["top_n"] = cfg.ACTIVE_LEARNING.HEURISTIC.TOP_N
        ret["impute_value"] = cfg.ACTIVE_LEARNING.HEURISTIC.IMPUTE_VALUE
        return ret

    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ) -> torch.Tensor:
        logger = logging.getLogger("activedet.acquisition.heuristics")
        scores = []
        for proba_map in proba_mapper:
            for image_id, MCproba_map in proba_map.items():
                preprocessed = [
                    self.preprocess_mc_instance(instance) for instance in MCproba_map
                ]
                mc_samples_entropy = [
                    self.merge_scores(entropy(instance))
                    for instance in preprocessed
                    if instance is not None
                ]
                mc_samples_entropy = (
                    torch.stack(mc_samples_entropy, dim=0)
                    if mc_samples_entropy
                    else self.impute_value
                )
                img_entropy = self.merge_scores(mc_samples_entropy)
                scores.append(img_entropy)
        scores = torch.tensor(scores)
        rank = self.rank_scores(scores)
        total_imputes = (scores == self.impute_value).sum()
        percent_impute = int(total_imputes/len(scores) * 100)
        logger.info(f"There are {total_imputes}/{len(scores)}={percent_impute} imputed values")
        logger.info(f"Predictions are: {scores[rank[:5]]}...{scores[rank[-5:]]}")

        if return_score:
            return rank, scores
        return rank

    def rank_scores(self, scores):
        # ranked by index
        return torch.argsort(scores, dim=-1, descending=True)

    def preprocess_mc_instance(self, instance):
        prob_scores = instance.prob_scores[instance.scores > self.scores_thresh]
        # If prob_scores is less than top_n, it gets all elements
        topn_prob_scores = prob_scores[: self.top_n] if self.top_n > 0 else prob_scores

        if topn_prob_scores.nelement() == 0:
            return None

        return topn_prob_scores


class QueryByCommittee(Heuristic):
    """Any aquisition function that obtains hypothesis of a group (committee)
    and measures disagreement among group
    """

    def __init__(self, cfg):
        pass

    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ) -> torch.Tensor:
        scores = []
        for proba_map in proba_mapper:
            for image_id, committee in proba_map.items():
                committee_distribution = [
                    self.merge_transform(member)
                    for member in committee
                    if self.is_valid(member)
                ]
                if committee_distribution == []:
                    break
                else:
                    committee_distribution = torch.stack(committee_distribution, dim=0)
                    disagreement_score = self.compute_disagreement(
                        committee_distribution
                    )
                    scores.append(disagreement_score)

        scores = torch.cat(scores, dim=0)
        ranks = self.rank(scores)
        if return_score:
            return ranks, scores
        return ranks

    def is_valid(self, member: Instances) -> bool:
        """Assess whether an instance can be filtered out or not
        This is where you'll be implement any filter mechanism according to your criteria

        Args:
            member : A committee member

        Returns:
            True if the member will stay. Otherwise, it will be removed.
        """
        return True

    def merge_transform(self, member: Instances) -> torch.Tensor:
        """A member's prediction is converted into a distribution usable for measuring disagreement
        It combines both merge and transform steps.
        Override this function if it requires some preprocessing steps
        Or, it requires the context of the whole Instances to calculate the transform.
        Args:
            member : a committee member's prediction. In detectron2, it is represented as Instances
        Returns:
            1D tensor distribution of size K usable for measuring disagreement
        """
        classifier = self.merge(member)
        distribution = self.transform(classifier)
        return distribution

    def merge(self, instance: Instances) -> torch.Tensor:
        """Merge an instance so that it becomes a classifier
        Args:
            instance : contains all detections predicted by a member

        Returns:
            a tensor representation of the instance

        Notes:
            An instance can have multiple detections. The aim of this step is to reduce detection
            into an output similar to a classifier. It is done to use in Query-By-Committee
        """
        return instance

    def transform(self, classifier: Union[Instances, torch.Tensor]) -> torch.Tensor:
        """Transforms classifier prediction into a distribution useful for measuring disagreement
        Args:
            classifier : representation of a classifier calculated by a member in a committee
        Returns:
            1D tensor of size K that will be utilized for measuring disagreement.
        """
        return classifier

    @abstractmethod
    def compute_disagreement(self, distributions: torch.Tensor) -> torch.Tensor:
        """Measure disagreement among the committee
        Args:
            distribution : N x K tensor that represents the votes of all classifiers for a single image
        Returns:
            disagreement scores in a 1D tensor of size 1
        """
        return NotImplemented

    def rank(self, scores: torch.Tensor, highestFirst=True) -> torch.Tensor:
        """Ranks disagreement scores
        Args:
            scores: a 1D tensor of size D that represents the score of an image.
        Returns:
            rank of the scores in descending order
        """
        return torch.argsort(scores, dim=-1, descending=highestFirst)


class VoteEntropy(QueryByCommittee):
    """Gets uncertainty through entropy from different classifiers
    Args:
         cfg = configuration set in launch.json the defines the merge mode
    Attributes:
         merge_scores = a merge of all scores/classifications of a certain image into one score/classification
         num_classes = the number of classifications in the classifier
         num_samples = the number of classifiers used in training

    """

    def __init__(self, cfg):
        self.merge_scores = merge_mode[cfg.ACTIVE_LEARNING.CLS_MERGE_MODE]
        assert (
            len(cfg["DATASETS"]["TEST"]) == 1
        ), "Vote Entropy assumes only one test set only!"
        self.num_classes = len(
            MetadataCatalog.get(cfg["DATASETS"]["TEST"][0]).thing_classes
        )
        self.num_samples = cfg.ACTIVE_LEARNING.POOL.MC_SIZE

    def is_valid(self, member: Instances) -> bool:
        return len(member.prob_scores > 0)

    def merge(self, instance: Instances) -> torch.Tensor:
        box_votes = []
        for j in range(self.num_classes):
            box_votes.append(instance.pred_classes.tolist().count(j))
        votes = torch.tensor(box_votes)
        votes = torch.argmax(votes)
        return votes

    def transform(self, classifier: Union[Instances, torch.Tensor]) -> torch.Tensor:
        return classifier

    def compute_disagreement(self, classifier_votes: torch.Tensor) -> torch.Tensor:
        uniques, count = torch.unique(classifier_votes, return_counts=True)
        vote_distribution = torch.true_divide(count, self.num_samples)
        # self.num_samples is meant to represent the number of classifiers used for training
        # this is under the assumption that all instances for one image per classifer has been merged into one classification
        return self.compute_cls_entropy(vote_distribution)

    def compute_cls_entropy(self, p):
        """
        Args:
            p = a tensor [IxV] that contains the vote distribution of each classification
                where:
                I - the image being classified (equal to 1)
                V - vote distribution/ratio of a classification (equal to the total amount of classifications)
        Returns:
            the 1 dimensional tensor for entropy
                where:
                entropy - 1 dimensional tensor that represents the entropy of the vote distribution of the image
        """
        entropy = -1 * torch.sum(p * torch.log2(p), dim=-1)
        return entropy[None, ...]


class ConsensusEntropy(QueryByCommittee):
    """Gets uncertainty through entropy from different classifiers
    Args:
         cfg = configuration set in launch.json the defines the merge mode
    Attributes:
         merge_scores = a merge of all scores/classifications of a certain image into one score/classification
         num_classes = the number of classifications in the classifier
         num_samples = the number of classifiers used in training

    """

    def __init__(self, cfg):
        self.merge_scores = merge_mode[cfg.ACTIVE_LEARNING.CLS_MERGE_MODE]
        assert (
            len(cfg["DATASETS"]["TEST"]) == 1
        ), "Vote Entropy assumes only one test set only!"
        self.num_classes = len(
            MetadataCatalog.get(cfg["DATASETS"]["TEST"][0]).thing_classes
        )
        self.num_samples = cfg.ACTIVE_LEARNING.POOL.MC_SIZE

    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ) -> torch.Tensor:
        scores = []
        vote_sum = []
        for proba_map in proba_mapper:
            for image_id, committee in proba_map.items():
                committee_distribution = [
                    self.merge_transform(member)
                    for member in committee
                    if self.is_valid(member)
                ]
                if committee_distribution == []:
                    break
                else:
                    committee_distribution = torch.sum(
                        torch.stack(committee_distribution), dim=0
                    )
                    # print(committee_distribution)
                    if vote_sum == []:
                        vote_sum = committee_distribution
                    else:
                        vote_sum = torch.vstack((vote_sum, committee_distribution))

        scores = self.compute_disagreement(vote_sum)
        ranks = torch.squeeze(self.rank(scores))
        if return_score:
            return ranks, scores
        return ranks

    def is_valid(self, member: Instances) -> bool:
        return len(member.prob_scores > 0)

    def merge(self, instance: Instances) -> torch.Tensor:
        detections_sum = torch.sum(instance.prob_scores, dim=0)
        prob_score_sum = torch.sum(detections_sum)
        votes = torch.true_divide(detections_sum, prob_score_sum)
        return votes

    def transform(self, classifier: Union[Instances, torch.Tensor]) -> torch.Tensor:
        return classifier

    def compute_disagreement(self, classifier_votes: torch.Tensor) -> torch.Tensor:
        vote_distribution = torch.true_divide(classifier_votes, self.num_samples)
        return self.compute_cls_entropy(vote_distribution)

    def compute_cls_entropy(self, p):
        """
        Args:
            p = a tensor [IxV] that contains the vote distribution of each classification
                where:
                I - the image being classified (equal to 1)
                V - vote distribution/ratio of a classification (equal to the total amount of classifications)
        Returns:
            the 1 dimensional tensor for entropy
                where:
                entropy - 1 dimensional tensor that represents the entropy of the vote distribution of the image
        """
        entropy = -1 * torch.sum(p * torch.log2(p), dim=-1)
        return entropy[None, ...]


class ClassificationCoreSet(Heuristic):
    """
    Implementation of kCenterGreedy from "Active learning for convolutional neural networks: A core-set approach"

    Steps:
        1. If center is none, initially select a random center
        2. Compute the distance of all points (centers + pool datapoints) to the pool datapoints
        3. Define the distance of a pool datapoint to be the smallest distance to a center
        4. Select the pool datapoint which has the maximum distance as a new center
        5. Repeat step 2 to 4 until the number of centers == ndata_to_label
        6. Label the new centers

    Attributes:
        ndata_to_label: number of centers to label
        center_features: contains the features each data point in the labeled dataset
            It is added by ActiveDatsetUpdater before calling this heuristic

    Reference:
        https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
    """

    @configurable
    def __init__(
        self,
        *,
        ndata_to_label: int,
        score_function: str = "max",
        impute_value: int = 1000,
        seed=None,
    ):
        # Used for instances that has no predictions
        # We don't discard because it ruins the order
        # We replace no predictions with the impute_value
        self.impute_value = torch.tensor([impute_value], dtype=torch.float)
        self.ndata_to_label = ndata_to_label
        self.score_function = dist_score_function[score_function.lower()]

        # Will be added by ActiveDatasetUpdater if active_learning.train_out_features has a valid name(s)
        # It will be continuously be updated for each new active step.
        self.center_features: Dict[str, torch.Tensors] = {}
        self.row_labeller: torch.Tensor = torch.empty(0)
        self.col_labeller: torch.Tensor = torch.empty(0)
        if seed is None:
            seed = comm.shared_random_seed()
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        assert comm.get_world_size() == 1, "CoreSet is only tested with Single GPU"

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["impute_value"] = cfg.ACTIVE_LEARNING.HEURISTIC.IMPUTE_VALUE
        ret["ndata_to_label"] = cfg.ACTIVE_LEARNING.NDATA_TO_LABEL
        return ret

    def initialize_centers(self, feature_map):
        # collect train features
        centers = torch.empty((0))
        if self.center_features:
            centers = torch.stack([value[0] for value in self.center_features.values()])
            self.row_labeller = torch.cat(
                (
                    torch.as_tensor([LabelCenterMode.CENTER] * len(centers)),
                    torch.as_tensor([LabelCenterMode.UNLABELLED] * len(feature_map)),
                ),
                dim=0,
            )
            self.col_labeller = torch.as_tensor(
                [LabelCenterMode.UNLABELLED] * len(feature_map)
            )
            dist_matrix = batched_all_pairs_l2_dist(
                torch.cat((centers, feature_map), dim=0)[None, ...],
                feature_map[None, ...],
            )
        else:
            idx = torch.randperm(len(feature_map), generator=self.generator)[:1]
            centers = feature_map[idx]
            self.row_labeller = torch.as_tensor(
                [LabelCenterMode.UNLABELLED] * len(feature_map)
            )
            self.row_labeller[idx] = LabelCenterMode.FOR_LABEL
            self.col_labeller = torch.as_tensor(
                [LabelCenterMode.UNLABELLED] * len(feature_map)
            )
            self.col_labeller[idx] = LabelCenterMode.FOR_LABEL
            dist_matrix = batched_all_pairs_l2_dist(
                feature_map[None, ...], feature_map[None, ...]
            )
        return centers, dist_matrix.squeeze()

    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ):
        """
        Args:
            proba_mapper: Generator yielding batches of images, where each image contains results from multiple MC Dropout runs.
                        This acts as the pool dataset for active learning.

        Returns:
            rank: A 1D tensor containing indices (with respect to the pool) arranged from the order in which the centers were selected.

        """

        features = []
        # collect features from generator
        # TODO: Convert into batch-style computation for improved memory efficiency
        for proba_map in proba_mapper:
            for image_id, feature in proba_map.items():
                features.append(torch.cat(feature, dim=0))

        # combine the list of feature tensors to one tensor
        features = torch.stack(features, dim=0)

        center_features, dist_matrix = self.initialize_centers(features)

        scores = []  # initialize with zeroes
        available_pool = sum(self.col_labeller == LabelCenterMode.UNLABELLED)
        max_label = min(self.ndata_to_label, available_pool)
        for l_ind in range(max_label):
            # Get all cluster centers
            row_mask = (self.row_labeller == LabelCenterMode.CENTER) | (
                self.row_labeller == LabelCenterMode.FOR_LABEL
            )
            col_mask = self.col_labeller != LabelCenterMode.FOR_LABEL
            center_to_pool = dist_matrix[row_mask][:, col_mask]
            min_distance_to_center, _ = torch.min(center_to_pool, dim=0)
            scores.append(min_distance_to_center)
            # add the image with the largest minimum distance to the list of centers
            ind = self.score_function(min_distance_to_center)

            true_col_ind = col_mask.nonzero()[ind]
            self.col_labeller[true_col_ind] = LabelCenterMode.FOR_LABEL
            # update new center features
            subrow = self.row_labeller[self.row_labeller != LabelCenterMode.CENTER]
            subrow[true_col_ind] = LabelCenterMode.FOR_LABEL
            self.row_labeller[self.row_labeller != LabelCenterMode.CENTER] = subrow

        rank = (self.col_labeller == LabelCenterMode.FOR_LABEL).nonzero().squeeze()

        if return_score:
            return rank, scores
        return rank


class KaoLT:
    """Localization Tightness Implementation of Kao et al

    FRCNN must be used for this heuristic
    Boxes and Proposal Boxes are required for calculation

    Uncertainty is measured as inconsistency between the classification score and the localization tightness score

    Kao_LT score

    The uncertainty score J for a detection Bj is given as
        J(Bj) = |T(Bj) + Pmax(Bj) − 1|
        where
            J(Bj) is the score of the box (detection),
            T(Bj) is the tightness component,

                T(Bj 0) = IoU(Bj, Rj)
                    Bj is the detection box
                    Rj is the proposal box

            Pmax is the max conf of the detection

    For an image with multiple detections, the image score is given as the minimum Kao_LT score
        TI(Ii) = minj J(Bj)

    """

    def __init__(self, cfg):
        # self.merge_scores = merge_mode[cfg.ACTIVE_LEARNING.CLS_MERGE_MODE]
        self.merge_scores = merge_mode[
            "min"
        ]  # must use min because, uncertainty score is used
        self.scores_thresh = cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH
        self.top_n = cfg.ACTIVE_LEARNING.HEURISTIC.TOP_N
        # assert cfg.ACTIVE_LEARNING.POOL.MC_SIZE == 1, "Monte Carlo Size must be set to 1"
        assert comm.get_world_size() == 1, "KaoLT is only tested with Single GPU"

    def __call__(
        self, proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None]
    ):
        scores = []
        for proba_map in proba_mapper:
            for image_id, MCproba_map in proba_map.items():
                mc_samples_entropy = []

                # MC_size should be equal to 1
                for instance in MCproba_map:
                    score_thresh_mask = instance.scores > self.scores_thresh
                    instance = instance[score_thresh_mask]

                    if len(instance) > self.top_n and self.top_n != 0:
                        instance = instance[: self.top_n]

                    if len(instance) > 0:
                        # Detection Uncertainty = J(Bj) = |T(Bj) +Pmax(Bj) − 1|
                        # Image Uncertainty = minj J(Bj)

                        mc_samples_entropy.append(
                            self.merge_scores(self.compute_detection_score(instance))
                        )

                # this merges the mc samples
                img_entropy = (
                    self.merge_scores(torch.Tensor(mc_samples_entropy))
                    if len(mc_samples_entropy) > 0
                    else 0
                )
                scores.append(img_entropy)

        scores = torch.Tensor(scores)
        rank = self.rank_scores(scores)

        return rank

    def compute_detection_score(self, instance):
        P_max, _ = torch.max(instance.prob_scores, -1)
        T = calculate_localization_tightness(
            instance.pred_boxes, instance.proposals.proposal_boxes
        )
        scores = abs(T + P_max - 1)
        return scores

    def rank_scores(self, scores):
        # ranked by index
        return torch.argsort(scores, dim=-1, descending=False)


class KaoLS:
    """Localization Stability From Kao et al, this will be called KaoLS

    The concept behind the localization stability is that, if the current model is stable to noise,
    meaning that the detection result does not dramatically change even if the input unlabeled image is corrupted by noise

    first the original image is used to obtain the Reference Boxes (detections in the original image)
    Gaussian Noise of different levels (standard deviation are added to the copy of the input image)
    for each noise level, the box with the highest IoU with the reference box Bj is obtained, it is denoted as Cj

    The localization stability score S_B for each reference box is denoted as
        S_B = sum_n(IoU(Bj,Cn(Bj 0)))/N
            where:
                N is the number of noise levels
                Cn(Bj) is the corresponding box, this is the detection in the noise image which has the highest IoU
                IoU is the intersection over union score

    The localization stability of the image S_I is given as
        S_I = sum(Pmax(B)*S_B(B))/sum(Pmax(B))

    """

    def __init__(self, cfg):
        # self.merge_scores = merge_mode[cfg.ACTIVE_LEARNING.CLS_MERGE_MODE]
        self.merge_scores = merge_mode[
            "min"
        ]  # must use min because uncertainty score is used
        self.scores_thresh = cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH
        self.top_n = cfg.ACTIVE_LEARNING.HEURISTIC.TOP_N
        self.sd_levels = cfg.ACTIVE_LEARNING.POOL.SD_LEVELS
        self.num_levels = len(self.sd_levels) - 1
        assert self.num_levels >= 1, "There should be at least 1 noise levels"
        assert comm.get_world_size() == 1, "KaoLS is only tested with Single GPU"

    def __call__(
        self, proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None]
    ):
        scores = torch.Tensor([])
        for proba_map in proba_mapper:
            for image_id, MCproba_map in proba_map.items():

                detection_across_levels = torch.Tensor([])

                # prepare reference instance
                ref_instance = MCproba_map[0]
                score_thresh_mask = ref_instance.scores > self.scores_thresh
                ref_instance = ref_instance[score_thresh_mask]

                if len(ref_instance.prob_scores) > self.top_n and self.top_n != 0:
                    ref_instance = ref_instance[: self.top_n]

                for instance in MCproba_map[1:]:
                    # for each prediction in noisy image
                    score_thresh_mask = instance.scores > self.scores_thresh
                    instance = instance[score_thresh_mask]

                    if len(instance.prob_scores) > self.top_n and self.top_n != 0:
                        instance = instance[: self.top_n]

                    if (
                        len(ref_instance.prob_scores) > 0
                        and len(instance.prob_scores) > 0
                    ):
                        # get score per detection (IoU(Bj,Cn(Bj 0)))/N
                        # this will be summed accross the image samples later
                        detection_candidate_IoU = (
                            torch.max(
                                pairwise_iou(
                                    ref_instance.pred_boxes, instance.pred_boxes
                                ),
                                -1,
                            ).values
                        ) / self.num_levels
                        if len(detection_across_levels) == 0:
                            detection_across_levels = detection_candidate_IoU.unsqueeze(
                                dim=0
                            )
                        else:
                            detection_across_levels = torch.cat(
                                (
                                    detection_across_levels,
                                    detection_candidate_IoU.unsqueeze(dim=0),
                                )
                            )

                if (
                    len(ref_instance.prob_scores) > 0
                    and len(detection_across_levels) > 0
                ):
                    # S_B = sum_n(IoU(Bj,Cn(Bj 0)))/N
                    S_B = torch.sum(detection_across_levels, 0)
                    # this merges the mc samples
                    # S_I = sum(Pmax(B)*S_B(B))/sum(Pmax(B))
                    P_max, _ = torch.max(ref_instance.prob_scores, -1)
                    P_max_sum = torch.sum(P_max)
                    S_I = torch.sum(P_max * S_B) / P_max_sum

                else:
                    # if there are no detections
                    S_I = torch.tensor(0.0)

                if len(scores) == 0:
                    scores = S_I.unsqueeze(dim=0)
                else:
                    scores = torch.cat((scores, S_I.unsqueeze(dim=0)))

        rank = self.rank_scores(scores)
        return rank

    def rank_scores(self, scores):
        # ranked by index
        return torch.argsort(scores, dim=-1, descending=False)


class KaoLS_C:
    """Localization Stability + C From Kao et al, this will be called KaoLS + C

    The concept behind the localization stability is that, if the current model is stable to noise,
    meaning that the detection result does not dramatically change even if the input unlabeled image is corrupted by noise

    first the original image is used to obtain the Reference Boxes (detections in the original image)
    Gaussian Noise of different levels (standard deviation are added to the copy of the input image)
    for each noise level, the box with the highest IoU with the reference box Bj is obtained, it is denoted as Cj

    The localization stability score S_B for each reference box is denoted as
        S_B = sum_n(IoU(Bj,Cn(Bj 0)))/N
            where:
                N is the number of noise levels
                Cn(Bj) is the corresponding box, this is the detection in the noise image which has the highest IoU
                IoU is the intersection over union score

    The localization stability of the image S_I is given as
        S_I = sum(Pmax(B)*S_B(B))/sum(Pmax(B))

    Image_Score = 1 - Pmax(B) - S_I

    """

    def __init__(self, cfg):
        # self.merge_scores = merge_mode[cfg.ACTIVE_LEARNING.CLS_MERGE_MODE]
        # must use min because uncertainty score is used
        self.merge_scores = merge_mode["min"]
        self.scores_thresh = cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH
        self.top_n = cfg.ACTIVE_LEARNING.HEURISTIC.TOP_N
        self.sd_levels = cfg.ACTIVE_LEARNING.POOL.SD_LEVELS
        self.num_levels = len(self.sd_levels) - 1
        assert self.num_levels >= 1, "There should be at least 1 noise levels"
        assert comm.get_world_size() == 1, "KaoLS_C is only tested with Single GPU"

    def __call__(
        self, proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None]
    ):
        scores = torch.Tensor([])
        for proba_map in proba_mapper:
            for image_id, MCproba_map in proba_map.items():

                detection_across_levels = torch.Tensor([])

                # prepare reference instance
                ref_instance = MCproba_map[0]
                score_thresh_mask = ref_instance.scores > self.scores_thresh
                ref_instance = ref_instance[score_thresh_mask]

                if len(ref_instance.prob_scores) > self.top_n and self.top_n != 0:
                    ref_instance = ref_instance[: self.top_n]

                for instance in MCproba_map[1:]:
                    # for each prediction in noisy image
                    score_thresh_mask = instance.scores > self.scores_thresh
                    instance = instance[score_thresh_mask]

                    if len(instance.prob_scores) > self.top_n and self.top_n != 0:
                        instance = instance[: self.top_n]

                    if (
                        len(ref_instance.prob_scores) > 0
                        and len(instance.prob_scores) > 0
                    ):
                        # get score per detection (IoU(Bj,Cn(Bj 0)))/N
                        # this will be summed accross the image samples later
                        detection_candidate_IoU = (
                            torch.max(
                                pairwise_iou(
                                    ref_instance.pred_boxes, instance.pred_boxes
                                ),
                                -1,
                            ).values
                        ) / self.num_levels
                        if len(detection_across_levels) == 0:
                            detection_across_levels = detection_candidate_IoU.unsqueeze(
                                dim=0
                            )
                        else:
                            detection_across_levels = torch.cat(
                                (
                                    detection_across_levels,
                                    detection_candidate_IoU.unsqueeze(dim=0),
                                )
                            )

                if (
                    len(ref_instance.prob_scores) > 0
                    and len(detection_across_levels) > 0
                ):
                    # S_B = sum_n(IoU(Bj,Cn(Bj 0)))/N
                    S_B = torch.sum(detection_across_levels, 0)
                    # this merges the mc samples
                    # S_I = sum(Pmax(B)*S_B(B))/sum(Pmax(B))
                    P_max, _ = torch.max(ref_instance.prob_scores, -1)
                    P_max_sum = torch.sum(P_max)
                    S_I = torch.sum(P_max * S_B) / P_max_sum

                    P_max_image = torch.max(ref_instance.prob_scores)

                else:
                    # if there are no detections
                    S_I = torch.tensor(0.0)
                    P_max_image = torch.tensor(0.0)

                image_score = torch.tensor(1) - P_max_image - S_I

                if len(scores) == 0:
                    scores = image_score.unsqueeze(dim=0)
                else:
                    scores = torch.cat((scores, image_score.unsqueeze(dim=0)))

        rank = self.rank_scores(scores)
        return rank

    def rank_scores(self, scores):
        # ranked by index
        return torch.argsort(scores, dim=-1, descending=True)


class CoreSet:
    """Implementation of kCenterGreedy from "Active learning for convolutional neural networks: A core-set approach"

    https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py

    Diversity based approach

    Distance is based on L2 Norm of the feature vector

    In this implementation. the most representative of the unlabeled pool is to be labelled

    """

    def __init__(self, cfg):
        # self.merge_scores = merge_mode[cfg.ACTIVE_LEARNING.CLS_MERGE_MODE]
        # self.merge_scores = merge_mode["min"] # must use min because uncertainty score is used
        # self.scores_thresh = cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH
        # self.top_n = cfg.ACTIVE_LEARNING.HEURISTIC.TOP_N
        self.N = cfg.ACTIVE_LEARNING.NDATA_TO_LABEL

        assert (
            cfg.ACTIVE_LEARNING.POOL.EVALUATOR == "BBEvaluator"
        ), "CoreSet will only work with the BBEvaluator"

        assert comm.get_world_size() == 1, "CoreSet is only tested with Single GPU"

    def __call__(
        self, proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None]
    ):

        features = []

        # collect features from generator
        for proba_map in proba_mapper:
            for _, feature in proba_map["features"].items():
                features.append(feature)

        n_pool = len(features)
        scores = torch.zeros(n_pool)  # initialize with zeroes
        features = torch.stack(
            features
        )  # combine the list of feature tensors to one tensor
        ind_to_label = []  # indices for the new batch of coreset

        ind_to_label.append(
            random.randrange(n_pool)
        )  # initialize the first center randomly

        for _ in range(self.N - 1):  # one is already labelled
            # update min distance to cluster centers
            center_features = features[ind_to_label]  # get features of the centers

            # compute the minimum distance of all the feature vectors of the pool images with respect to the centers
            # centers are expected to have a min distance of 0
            min_distance_to_center, _ = torch.min(
                torch.cdist(features, center_features), axis=-1
            )

            # add the image with the largest minimum distance to the list of centers
            ind_to_label.append(int(torch.argmax(min_distance_to_center)))

        assert len(set(ind_to_label)) == len(ind_to_label), "the indices must be unique"

        for i in ind_to_label:
            # to comply with the framework, the centers will be given a high score
            scores[i] = torch.tensor(1000)

        # sort by descending order
        rank = self.rank_scores(scores)
        return rank

    def rank_scores(self, scores):
        # ranked by index
        return torch.argsort(scores, dim=-1, descending=True)


class YooLearningLoss:
    """Implementation of Learning Loss applied in Object Detection
    The original Learning Loss is applied in an image-level (not instance).
    It only captures the highest predicted loss according to the loss prediction module
    This heuristic assumes that the conifuration would have an auxilliary module
    that predicts expected training loss

    Paper: `Learning Loss for Active Learning` By Yoo and Kweon (2019)
    https://arxiv.org/pdf/1905.03677.pdf

    Notes:
        It is now compatible with distributed training
    """

    @configurable
    def __init__(self, *, impute_value=1000):

        # Used for instances that has no predictions
        # We don't discard because it ruins the order
        # We replace no predictions with the impute_value
        self.impute_value = torch.tensor(impute_value, dtype=torch.float)

    @classmethod
    def from_config(cls, cfg):
        assert (
            cfg.ACTIVE_LEARNING.POOL.SAMPLING_METHOD == "Trivial"
        ), "The original Learning Loss contains one prediction/instance only"

        ret = {}
        ret["impute_value"] = cfg.ACTIVE_LEARNING.HEURISTIC.IMPUTE_VALUE
        return ret

    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ):
        logger = logging.getLogger("activedet.acquisition.heuristics")
        # In this case, each preds is a dictionary of batch prediction
        pred_losses = []
        for preds in proba_mapper:
            logger.info("Calculating heuristics...")
            for i, pred_rank in enumerate(comm.gather(preds)):
                logger.info(
                    f"rank {i}: Current image in batch, {[pred for pred in pred_rank]}"
                )

            reduced = self.reduce_batch(preds)
            reduced_all = torch.cat(comm.all_gather(reduced), dim=0)

            pred_losses.append(reduced_all)

        pred_losses = torch.cat(pred_losses, dim=0)
        logger.info(f"First 10 prediction, {[p for p in pred_losses[:10]]}")

        rank = torch.argsort(pred_losses, dim=0, descending=True)
        if return_score:
            return rank, pred_losses
        return rank

    def reduce_batch(self, batch: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """Describes on how to reduce the batch of prediction
        Args:
            batch: contains the batch prediction in Instances-style format
                - key(str) : "image_id"
                - value(List[torch.Tensor]): Prediction
        Returns:
            loss predictions on a given instance batch in 1D tensor of size B
        Notes:
            Let B = batch size of the dictionary, i.e. len(batch) == B
        """
        # In pred_loss, we only use the first element as the succeeding elements are always redundant
        pred_batch = [
            pred[0].pred_loss[0] if len(pred[0]) else self.impute_value
            for image_id, pred in batch.items()
        ]
        return torch.stack(pred_batch, dim=0).float()


class Discrepancy:
    """Implementation of Discrepancy as the acquisition function for active learning.
    This is described in the paper `Multiple Instance Active Learning for Object Detection`
    In the paper, it is also known informative uncertainty selection

    Note:
        Monte Carlo Estimation is not supported for this heuristic. On a repeated inference setting,
        it will only take first prediction across samples
    """

    def __init__(self, *, merge="none", threshold=0, top_n=0, impute_value=1000):
        self.merge_scores = merge_mode[merge.lower()]
        self.scores_thresh = threshold
        self.top_n = top_n
        self.impute_value = impute_value

    def __call__(
        self,
        proba_mapper: Generator[Dict[str, List[torch.Tensor]], None, None],
        return_score=False,
    ) -> torch.Tensor:
        logger = logging.getLogger("activedet.acquisition.heuristics")

        scores = []
        for preds in proba_mapper:
            mini_batch = self.reduce_batch(preds).cpu()
            batch = torch.cat(comm.all_gather(mini_batch), dim=0)
            scores.append(batch)

        # Expected Shape: nPool
        pred_uncertainty = torch.cat(scores, dim=0)

        logger.info(f"First 10 prediction, {[p for p in pred_uncertainty[:10]]}")

        rank = torch.argsort(pred_uncertainty, dim=0, descending=True)
        if return_score:
            return rank, pred_uncertainty
        return rank

    def reduce_batch(self, batch_instance):

        # Expected Shape: List[torch.Tensor]: Total length is batch
        #   Each item will have a shape: R x K, where R is total regions across all pyramid levels
        #   Each R for each level would be different

        pred_batch = [
            self.extract_uncertainty(pred[0]) if len(pred[0]) else self.impute_value
            for image_id, pred in batch_instance.items()
        ]
        return torch.stack(pred_batch,dim=0)

    def uncertainty(self, instance):
        # R x K
        disc_l2 = (instance.proba_scores1 - instance.proba_scores2).pow(2).squeeze()
        return disc_l2

    def extract_uncertainty(self, instance: Instances) -> torch.Tensor:
        """Calculates the uncertainty for a given instance object
        Args:
            instance (Instances): describes all instance in a single image
                It expected to contain an attribute of uncertainty with
                    shape (R, K)
        Returns:
            a scalar Tensor with shape () that describes the image-level score
        """
        # Expected shape: (R, )
        uncertainty_regions = self.merge_scores(self.uncertainty(instance), dim=1)
        # sort prediction starting with the highest uncertainty
        values, _ = torch.sort(uncertainty_regions, descending=True)
        # Gather the top_n uncertainty and take its average
        return self.merge_scores(values[: self.top_n])


def build_heuristic(cfg):
    if cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "SemSegHeuristic":
        return SemSegHeuristic(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "Random":
        return RandomHeuristic(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "ClassificationEntropy":
        return ClassificationEntropy(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "ClsCoreSet":
        return ClassificationCoreSet(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "KaoLT":
        return KaoLT(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "KaoLS":
        return KaoLS(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "KaoLS_C":
        return KaoLS_C(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "CoreSet":
        return CoreSet(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "YooLearningLoss":
        return YooLearningLoss(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "VoteEntropy":
        return VoteEntropy(cfg)
    elif cfg.ACTIVE_LEARNING.HEURISTIC.NAME == "ConsensusEntropy":
        return ConsensusEntropy(cfg)
    else:
        raise NotImplementedError(
            f"The Heurisitic {cfg.ACTIVE_LEARNING.HEURISITIC.NAME} is not implemented!"
        )
