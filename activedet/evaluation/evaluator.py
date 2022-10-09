from contextlib import contextmanager
from typing import Dict
from operator import itemgetter
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from pathlib import Path

from activedet.utils.math import cosine_distance, normalize
from activedet.evaluation.custom_voc_eval import fig_to_np

PROJECT_DIR = Path(__file__).parent.parent


@contextmanager
def validation_context(model):
    """
    Temporarily set the model into training mode in order
    to calculate the validation loss. It is restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.train()
    yield
    model.train(training_mode)


@contextmanager
def device_cpu_context(model):
    device = next(model.parameters()).device
    model.to("cpu")
    yield device
    model.to(device)


@contextmanager
def device_gpu_context(model, device="cuda"):
    orig_device = next(model.parameters()).device
    model.to(device)
    yield
    model.to(orig_device)


class ClassificationAccuracy(DatasetEvaluator):
    def reset(self):
        self.corr = self.total = 0

    def process(self, inputs, outputs):
        label = torch.tensor(
            [a["category_id"] for anno in inputs for a in anno["annotations"]]
        )
        pred_output = torch.cat(
            [instance["instances"].pred_classes.cpu() for instance in outputs]
        )
        self.corr += (pred_output == label.cpu()).sum().item()
        self.total += len(label)

    def evaluate(self):
        all_corr_total = comm.all_gather([self.corr, self.total])
        corr = sum(x[0] for x in all_corr_total)
        total = sum(x[1] for x in all_corr_total)
        return {"accuracy": corr / total}

class SemanticEmbeddingEvaluator(DatasetEvaluator):
    def __init__(self, eval_dataset: str):
        self.emb_matrix = torch.load(
            str(PROJECT_DIR / "data/{}_vec.pth".format(eval_dataset))
        )
        self.emb_matrix = normalize(self.emb_matrix)

    def reset(self):
        self.corr = self.corr3 = self.corr5 = self.total = 0
        self.incorrect_pred = defaultdict(list)

    def process(self, inputs, outputs):
        label = torch.tensor(
            [a["category_id"] for anno in inputs for a in anno["annotations"]]
        )
        # Expected Shape: B x dim
        pred_output = torch.cat(
            [instance["instances"].pred_emb for instance in outputs], dim=0
        )
        emb_matrix = self.emb_matrix.to(pred_output.device)
        # Expected shape: BatchSize x num_classes
        sim_matrix = cosine_distance(pred_output, emb_matrix).cpu().detach()
        label = label.cpu().detach()
        miss_idx = sim_matrix.argmax(dim=1) != label
        for idx in miss_idx.nonzero():
            similarity_vector = sim_matrix[idx]
            image = inputs[idx.item()]["image"].cpu().detach().to(dtype=torch.uint8)
            pred = similarity_vector.argmax(dim=1).cpu().item()
            gt = label[idx].cpu().item()
            self.incorrect_pred[pred].append(
                {
                    "similarity": similarity_vector.cpu().detach(),
                    "image": image,
                    "ground": gt,
                }
            )
        self.corr += (sim_matrix.argmax(dim=1) == label).sum().cpu().item()
        self.corr3 += (
            (label.unsqueeze(dim=1) == sim_matrix.topk(k=3, dim=1)[1])
            .sum()
            .cpu()
            .item()
        )
        self.corr5 += (
            (label.unsqueeze(dim=1) == sim_matrix.topk(k=5, dim=1)[1])
            .sum()
            .cpu()
            .item()
        )
        self.total += len(label)

    def evaluate(self):
        all_corr_total = comm.all_gather(
            [self.corr, self.total, self.corr3, self.corr5, self.incorrect_pred]
        )
        corr = sum(x[0] for x in all_corr_total)
        corr3 = sum(x[2] for x in all_corr_total)
        corr5 = sum(x[3] for x in all_corr_total)
        total = sum(x[1] for x in all_corr_total)
        incorrect_preds = defaultdict(list)
        [incorrect_preds[keys].extend(value) for x in all_corr_total for keys, value in x[4].items()]
        del self.incorrect_pred
        del all_corr_total
        plots = {}
        num_classes = self.emb_matrix.shape[0]
        x = np.arange(num_classes)
        freq = np.zeros(num_classes)
        for key, attributes in incorrect_preds.items():
            freq[key] = len(attributes) / total
            ranks = torch.argsort(torch.stack([item["similarity"].max() for item in attributes],dim=0))
            if len(ranks) > 5:
                top5 = [attr["image"] for attr in itemgetter(*ranks[:5].tolist())(attributes)]
                bottom5 = [attr["image"] for attr in itemgetter(*ranks[-5:].tolist())(attributes)]
                plots["class_{}_bottom5".format(key)] = make_grid(bottom5,nrow=1)
            else:
                top5 = [attr["image"] for attr in attributes]

            plots["class_{}_top5".format(key)] = make_grid(top5, nrow=1)            

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(x,freq)

        arr = fig_to_np(fig, ax)
        plots["mistake distribution"] = arr
        
        return {
            "accuracy": corr / total,
            "top3": corr3 / total,
            "top5": corr5 / total,
            "plot": plots,
        }
