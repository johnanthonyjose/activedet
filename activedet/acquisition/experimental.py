from typing import Generator, Dict, List
from collections import defaultdict
import logging
from detectron2.config.config import configurable
import torch
import numpy as np
import detectron2.utils.comm as comm
from .heuristics import entropy, Heuristic, merge_mode


class ConfidenceBinnedEntropy(Heuristic):
    """Bin all top1 predictions into 0.1 resolution. Then, for each resolution, sort it according to top1 entropy
    Sort the resolution according to lower confidence first.
    """

    @configurable
    def __init__(
        self, *, merge="none", threshold=0, resolution=0.1, top_n=0, impute_value=1000
    ):
        self.merge_scores = merge_mode[merge.lower()]
        self.scores_thresh = threshold
        self.bin = np.array([i for i in np.arange(threshold, 1.0, resolution)])
        if threshold > 0:
            self.bin = np.insert(self.bin,0,0.0,axis=0)
        self.bin = torch.from_numpy(self.bin)
        self.top_n = top_n
        # Used for instances that has no predictions
        # We don't discard because it ruins the order
        # We replace no predictions with the impute_value
        self.impute_value = torch.tensor([impute_value], dtype=torch.float)
        assert (
            comm.get_world_size() == 1
        ), "ConfidenceBinnedEntropy is only tested with Single GPU"

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
        binner = defaultdict(list)
        # Proba_map is a batch of images
        idx = 0
        for proba_map in proba_mapper:
            # MCproba_map is a single image with multiple MonteCarlo Predictions
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
                pred_score = [
                        # Get the confidence score
                        self.merge_scores(instance.max(dim=-1)[0])
                        for instance in preprocessed
                        if instance is not None
                    ]
                pred_score = torch.stack(pred_score,dim=0) if pred_score else torch.tensor([0.0])
                single_ent = self.merge_scores(mc_samples_entropy)
                single_score = self.merge_scores(pred_score)
                # Get the highest threshold
                ind = (single_score >= self.bin).nonzero().max()
                # Bin its entropy
                pred = {"name": image_id, "entropy": single_ent, "id": idx}
                binner[self.bin[ind].item()].append(pred)
                idx += 1

        # After binning all the predictions, we sort out each bin according to the highest entropy
        binner = {
            thresh: sorted(preds_list, key=lambda x: x["entropy"], reverse=True)
            for thresh, preds_list in binner.items()
        }

        # Now, we want to calculate the rank of each image.
        # Recall, for instance, we have the ff:
        # index according to the proba_map: 0, 1, 2, 3, 4
        # rank according to heuristic:      3, 4, 1, 2, 0
        # What this means is that first image should be at rank 3 (start counting from 0).
        # We assume that lowest confidence would be in the first amoung all other bins
        rank = torch.tensor(
            [im_desc["id"] for thresh, preds in sorted(binner.items()) for im_desc in preds]
        )
        scores = torch.tensor(
            [
                im_desc["entropy"]
                for thresh, preds in sorted(binner.items())
                for im_desc in preds
            ]
        )

        total_imputes = (scores == self.impute_value).sum()
        percent_impute = int(total_imputes/len(scores) * 100)
        logger.info(f"There are {total_imputes}/{len(scores)}={percent_impute} imputed values")
        logger.info(f"Predictions are: {scores[:5]}...{scores[-5:]}")

        if return_score:
            return rank, scores
        return rank

    def preprocess_mc_instance(self, instance):
        prob_scores = instance.prob_scores[instance.scores > self.scores_thresh]
        # If prob_scores is less than top_n, it gets all elements
        topn_prob_scores = prob_scores[: self.top_n] if self.top_n > 0 else prob_scores

        if topn_prob_scores.nelement() == 0:
            return None

        return topn_prob_scores
