import torch
import pytest
from detectron2.structures import Instances
from detectron2.config import get_cfg
from activedet.config import add_active_learning_config
from activedet.acquisition.heuristics import ClassificationEntropy, VoteEntropy, ConsensusEntropy, YooLearningLoss, ClassificationCoreSet
from activedet.acquisition.experimental import ConfidenceBinnedEntropy
class DummyInstances:
    def __init__(self, prob_scores):
        self.prob_scores = torch.tensor(prob_scores)
        self.scores = torch.tensor([])
        self.pred_classes = torch.tensor([])
        if len(self.prob_scores) > 0:
            self.scores, self.pred_classes = torch.max(self.prob_scores, dim=-1)

@pytest.fixture
def dummydata1_batches():
    """Inspired from modAL Vote entropy
    total number of batches = 5
    batch_size = 1
    MC_size = 3
    num_instances per Instance = 1
    number of classes (K) = 3
    https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#vote-entropy
    """
    return [
        {   # there are two votes for 0, one votes for 1 and zero votes for 2
            "img_1": [  #class: 0    1     2        
                DummyInstances([[0.8, 0.1, 0.0]]),  # \
                DummyInstances([[0.3, 0.7, 0.0]]),  # |
                DummyInstances([[1.0, 0.0, 0.0]]),  # |  <-- class probabilities for the first classifier
            ]                                       # | 
        },                                          # /
        {
            "img_2": [
                DummyInstances([[0.0, 1.0, 0.0]]),  # \
                DummyInstances([[0.4, 0.6, 0.0]]),  # |
                DummyInstances([[0.0, 0.0, 1.0]]),  # |  <-- class probabilities for the second classifier
            ]                                       # |  
        },                                          # /
        {
            "img_3": [
                DummyInstances([[0.7, 0.2, 0.1]]),  # \
                DummyInstances([[0.4, 0.0, 0.6]]),  # |
                DummyInstances([[0.3, 0.5, 0.2]]),  # |  <-- class probabilities for the third classifier
            ]                                       # |  
        },                                          # /
        {
            "img_4": [
                DummyInstances([[0.0, 0.0, 1.0]]),  # \
                DummyInstances([[0.2, 0.3, 0.5]]),  # |
                DummyInstances([[0.1, 0.1, 0.8]]),  # |  <-- class probabilities for the fourth classifier
            ]                                       # |  
        },                                          # /
        {
            "img_5": [
                DummyInstances([[0.0, 1.0, 0.0]]),  # \
                DummyInstances([[0.2, 0.3, 0.5]]),  # |
                DummyInstances([[0.1, 0.1, 0.8]]),  # |  <-- class probabilities for the fourth classifier
            ]                                       # |  
        },                                          # /

    ]

@pytest.fixture
def dummydata2_batches():
    """Inspired from modAL Consensus entropy
    https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#disagreement-sampling
    """
    return [
        {
            "img_1": [
                DummyInstances([[0.8, 0.1, 0.0]]),  # \
                DummyInstances([[0.3, 0.7, 0.0]]),  # |
                DummyInstances([[1.0, 0.0, 0.0]]),  # |  <-- class probabilities for the first classifier
                DummyInstances([[0.2, 0.2, 0.6]]),  # |
                DummyInstances([[0.2, 0.7, 0.1]]),  # |
            ]                                       # | 
        },                                          # /
        {
            "img_2": [
                DummyInstances([[0.0, 1.0, 0.0]]),  # \
                DummyInstances([[0.4, 0.6, 0.0]]),  # |
                DummyInstances([[0.2, 0.7, 0.1]]),  # |  <-- class probabilities for the second classifier
                DummyInstances([[0.3, 0.1, 0.6]]),  # |
                DummyInstances([[0.0, 0.0, 1.0]]),  # |
            ]                                       # |  
        },                                          # /
        {
            "img_3": [
                DummyInstances([[0.7, 0.2, 0.1]]),  # \
                DummyInstances([[0.4, 0.0, 0.6]]),  # |
                DummyInstances([[0.3, 0.2, 0.5]]),  # |  <-- class probabilities for the third classifier
                DummyInstances([[0.1, 0.0, 0.9]]),  # |
                DummyInstances([[0.0, 0.1, 0.9]]),  # |
            ]                                       # |  
        },                                          # /
    ]

@pytest.fixture
def dummydata3_batches():
    """Inspired from modAL Consensus entropy

    Re-arranges the dummydata2_batches in order to comply on how modAL calculated its consensus entropy
    https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#vote-entropy
    """
    return [
        {   # there are two votes for 0, one votes for 1 and zero votes for 2
            "img_1": [  #class: 0    1     2        
                DummyInstances([[0.8, 0.1, 0.0]]),  # \
                DummyInstances([[0.0, 1.0, 0.0]]),  # |
                DummyInstances([[0.7, 0.2, 0.1]]),  # |  <-- class probabilities for the first classifier
            ]                                       # | 
        },                                          # /
        {
            "img_2": [
                DummyInstances([[0.3, 0.7, 0.0]]),  # \
                DummyInstances([[0.4, 0.6, 0.0]]),  # |
                DummyInstances([[0.4, 0.0, 0.6]]),  # |  <-- class probabilities for the second classifier
            ]                                       # |  
        },                                          # /
        {
            "img_3": [
                DummyInstances([[1.0, 0.0, 0.0]]),  # \
                DummyInstances([[0.2, 0.7, 0.1]]),  # |
                DummyInstances([[0.3, 0.2, 0.5]]),  # |  <-- class probabilities for the third classifier
            ]                                       # |  
        },                                          # /
        {
            "img_4": [
                DummyInstances([[0.2, 0.2, 0.6]]),  # \
                DummyInstances([[0.3, 0.1, 0.6]]),  # |
                DummyInstances([[0.1, 0.0, 0.9]]),  # |  <-- class probabilities for the fourth classifier
            ]                                       # |  
        },                                          # /
        {
            "img_5": [
                DummyInstances([[0.2, 0.7, 0.1]]),  # \
                DummyInstances([[0.0, 0.0, 1.0]]),  # |
                DummyInstances([[0.0, 0.1, 0.9]]),  # |  <-- class probabilities for the fourth classifier
            ]                                       # |  
        },                                          # /

    ]


@pytest.fixture
def instance_batch_trivial():
    return [
        { 
            "img_1": [      
                Instances(image_size=(50,50),
                    prob_scores=torch.tensor([[0.8, 0.1, 0.1],
                                              [0.3, 0.3, 0.4],
                                              [0.25,0.5, 0.25]]),
                    scores=torch.tensor([0.8, 0.4, 0.5]),
                    pred_loss=torch.tensor([0.5,0.5,0.5])
                    ),
            ]
        },
        { 
            "img_2": [   
                Instances(image_size=(50,50), 
                    prob_scores=torch.tensor([[0.9, 0.05, 0.05],
                                              [0.5, 0.1, 0.1],
                                              [0.2,0.1, 0.7]]),
                    scores=torch.tensor([0.9, 0.5, 0.7]),
                    pred_loss=torch.tensor([-0.5,-0.5,-0.5])
                    ),
            ]
        },
        { 
            "img_3": [
                Instances(image_size=(50,50), 
                    prob_scores=torch.tensor([[0.9, 0.05, 0.05],
                                              [0.5, 0.1, 0.1],
                                              [0.2,0.1, 0.7]]),
                    scores=torch.tensor([0.3, 0.2, 0.5]),
                    pred_loss=torch.tensor([10,10,10])
                    ),
            ]
        },
        { 
            "img_4": [
                Instances(image_size=(50,50), 
                    prob_scores=torch.tensor([[0.3, 0.3, 0.4],
                                              [0.15, 0.05, 0.8],
                                              [0.05,0.55, 0.4]]),
                    scores=torch.tensor([0.4, 0.8, 0.55]),
                    pred_loss=torch.tensor([30,30,30])
                    ),
            ]
        },
        {
            "img_5": [
                Instances(image_size=(50,50), 
                    prob_scores=torch.tensor([[0.1, 0.5, 0.4],
                                              [0.6, 0.05, 0.35],
                                              [0.2,0.4, 0.4]]),
                    scores=torch.tensor([0.5, 0.6, 0.4]),
                    pred_loss=torch.tensor([20,20,20])
                    ),
            ]
        },
    ]

@pytest.fixture
def multi_instance_batch_trivial():
    return [
        { 
            "img_1": [      
                Instances(image_size=(50,50), 
                    pred_scores=torch.tensor([[0.8, 0.1, 0.0],[0.8, 0.1, 0.0]]),
                    pred_loss=torch.tensor([0.5,0.5])
                    ),
            ],
            "img_2": [   
                Instances(image_size=(50,50), 
                    pred_scores=torch.tensor([[0.3, 0.7, 0.0],[0.3, 0.7, 0.0],[0.3, 0.7, 0.0]]),
                    pred_loss=torch.tensor([-0.5,-0.5,-0.5])
                    ),
            ]
        },
        { 
            "img_3": [
                Instances(image_size=(50,50), 
                    pred_scores=torch.tensor([[0.3, 0.2, 0.5]]),
                    pred_loss=torch.tensor([10])
                    ),
            ],
            "img_4": [
                Instances(image_size=(50,50), 
                    pred_scores=torch.tensor([[0.1, 0.0, 0.9],[0.1, 0.0, 0.9]]),
                    pred_loss=torch.tensor([30,30])
                    ),
            ],
            "img_5": [
                Instances(image_size=(50,50), 
                    pred_scores=torch.tensor([[0.0, 0.0, 1.0],[0.0, 0.0, 1.0]]),
                    pred_loss=torch.tensor([20,20])
                    ),
            ]
        },
    ]

@pytest.fixture
def features_batch_trivial():
    return [
        {
            "img_1": [
                torch.tensor([0.3, 0.7, 0.0]),
            ]
        },
        { 
            "img_2": [
                torch.tensor([0.3, 0.2, 0.5]),
            ]
        },
        { 
            "img_3": [
                torch.tensor([0.1, 0.0, 0.9]),
            ]
        },
        
    ]

@pytest.fixture
def cfg():
    config = get_cfg()
    add_active_learning_config(config)
    return config

def test_ClassificationEntropy_configurable(cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    assert ClassificationEntropy(cfg)

def test_ClassificationEntropy_no_config():
    assert ClassificationEntropy(merge="max",threshold=0.05,top_n=50, impute_value=0)

def test_ClassificationEntropy_1(dummydata1_batches, cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.POOL.MC_SIZE = 3
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "mean"
    heuristic = ClassificationEntropy(cfg)
    ranks, scores = heuristic(dummydata1_batches, return_score=True)
    assert len(ranks) == 5 # There's a total of 5 instances in all of the batches
    assert torch.allclose(scores, torch.tensor([0.3199, 0.2243, 0.8348,0.5562,0.5562]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([2     , 3     , 4     , 0    , 1     ])), f"Calculated rank: {ranks}"

def test_ClassificationEntropy_2(dummydata2_batches, cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.POOL.MC_SIZE = 5
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "mean"
    heuristic = ClassificationEntropy(cfg)
    ranks, scores = heuristic(dummydata2_batches, return_score=True)
    assert len(ranks) == 3 # There's a total of 5 instances in all of the batches
    assert torch.allclose(scores, torch.tensor([0.5423, 0.4745, 0.6309]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([2     , 0     , 1     ])), f"Calculated rank: {ranks}"

def test_ClassificationEntropy_instancebatch(instance_batch_trivial, cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "mean"
    heuristic = ClassificationEntropy(cfg)
    ranks, scores =  heuristic(instance_batch_trivial, return_score=True)
    assert len(ranks) == 5
    assert torch.allclose(scores, torch.tensor([0.9226, 0.6642, 0.6642, 0.8490, 0.9407]),atol=1e-4), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([4     , 0     , 3      ,1       , 2     ])), f"Calculated rank: {ranks}"

def test_ClassificationEntropy_p90(instance_batch_trivial, cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "mean"
    cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH = 0.90
    heuristic = ClassificationEntropy(cfg)
    ranks, scores =  heuristic(instance_batch_trivial, return_score=True)
    assert len(ranks) == 5
    assert torch.allclose(scores, torch.tensor([1000., 1000., 1000., 1000., 1000.]),atol=1e-4), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([0     , 1     , 2      ,3       , 4     ])), f"Calculated rank: {ranks}"

def test_ConfidenceBinnedEntropy_instancebatch(instance_batch_trivial, cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "mean"
    heuristic = ConfidenceBinnedEntropy(cfg)
    ranks, scores =  heuristic(instance_batch_trivial, return_score=True)
    assert len(ranks) == 5
    
    assert torch.allclose(scores, torch.tensor([0.9226, 0.8490, 0.6642, 0.6642, 0.9407]),atol=1e-4), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([0     , 3     , 1      ,2       , 4     ])), f"Calculated rank: {ranks}"

def test_ConfidenceBinnedEntropy_instancebatch_p50(instance_batch_trivial, cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/classification_entropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "mean"
    cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH = 0.50
    heuristic = ConfidenceBinnedEntropy(cfg)
    ranks, scores =  heuristic(instance_batch_trivial, return_score=True)
    assert len(ranks) == 5
    
    assert torch.allclose(scores, torch.tensor([1000., 0.8237, 0.7289, 0.5981, 0.6390]),atol=1e-4), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([2     , 4     , 3     , 1      ,0     ])), f"Calculated rank: {ranks}"


def test_VoteEntropy_modAL(dummydata1_batches, cfg):
    """This test is based on Vote Entropy of modAL.
    The difference is that modAL uses log10. In our case, we use log2
    https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#vote-entropy
    """
    cfg.merge_from_file("./configs/PascalVOC-Detection/VoteEntropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.POOL.MC_SIZE = 3
    vote_heuristic = VoteEntropy(cfg)
    ranks, scores = vote_heuristic(dummydata1_batches, return_score=True)
    assert len(ranks) == 5 # There's a total of 5 instances in all of the batches
    assert torch.allclose(scores, torch.tensor([0.9182, 0.9182, 1.5849, 0.0, 0.9182]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([2     , 0     , 1     , 4  , 3     ])), f"Calculated rank: {ranks}"


def test_VoteEntropy_tony(dummydata2_batches, cfg):
    """This is based on manual computation of tony according to modAL calculation
    """
    cfg.merge_from_file("./configs/PascalVOC-Detection/VoteEntropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.POOL.MC_SIZE = 5
    vote_heuristic = VoteEntropy(cfg)
    ranks, scores = vote_heuristic(dummydata2_batches, return_score=True)
    assert len(ranks) == 3 # There's a total of 3 instances in all of its batches
    assert torch.allclose(scores, torch.tensor([1.5219, 0.9709, 0.7219]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([0     , 1     , 2     ])), f"Calculated rank: {ranks}"


def test_ConsensusEntropy_carl(dummydata3_batches, cfg):
    """This is based on manual computation of Carl according to modAL calculation
    """
    cfg.merge_from_file("./configs/PascalVOC-Detection/ConsensusEntropy_faster_rcnn.yaml")
    cfg.ACTIVE_LEARNING.POOL.MC_SIZE = 3
    consensus_heuristic = ConsensusEntropy(cfg)
    ranks, scores = consensus_heuristic(dummydata3_batches, return_score=True)
    assert len(ranks) == 5 # There's a total of 3 instances in all of its batches
    assert torch.allclose(scores, torch.tensor([1.1711, 1.5179, 1.4855, 1.1568, 1.1589]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([1     , 2     ,0      ,4      ,3     ])), f"Calculated rank: {ranks}"

def test_CoreSet(cfg, features_batch_trivial):
    heuristic = ClassificationCoreSet(
        impute_value=1000,
        ndata_to_label= 3
        )

    heuristic.center_features = {
            "center_1": [
                torch.tensor([0.8, 0.1, 0.0]),
            ],
            "center_2": [
                torch.tensor([0.0, 0.0, 1.0]),
            ]
        }        
    expected_scores = [
            torch.tensor([0.7810, 0.6164, 0.1414]),
            torch.tensor([        0.6164, 0.1414]),
            torch.tensor([                0.1414])
        ]                                
    ranks, scores =  heuristic(features_batch_trivial, return_score=True)
    assert len(ranks) == 3, "output shape not ok"
    for actual, expected in zip(scores, expected_scores):
        assert torch.allclose(actual, expected,atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([0     , 1     ,2])), f"Calculated rank: {ranks}"

def test_LearnLoss_configurable(cfg):
    cfg.merge_from_file("./configs/PascalVOC-Detection/LearnLoss_faster_rcnn.yaml")
    assert YooLearningLoss(cfg)

def test_LearnLoss_no_config():
    assert YooLearningLoss(impute_value=0)

def test_LearnLoss(cfg, instance_batch_trivial):
    cfg.merge_from_file("./configs/PascalVOC-Detection/LearnLoss_faster_rcnn.yaml")
    heuristic = YooLearningLoss(cfg)
    ranks, scores =  heuristic(instance_batch_trivial, return_score=True)
    assert len(ranks) ==5
    assert torch.allclose(scores, torch.tensor([0.5, -0.5, 10, 30, 20]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([3     , 4     ,2      ,0      ,1     ])), f"Calculated rank: {ranks}"

def test_LearnLoss_multi_instance(cfg, multi_instance_batch_trivial):
    cfg.merge_from_file("./configs/PascalVOC-Detection/LearnLoss_faster_rcnn.yaml")
    heuristic = YooLearningLoss(cfg)
    ranks, scores =  heuristic(multi_instance_batch_trivial, return_score=True)
    assert len(ranks) ==5
    assert torch.allclose(scores, torch.tensor([0.5, -0.5, 10, 30, 20]), atol=1e-04), f"Calculated Score: {scores}"
    assert torch.equal(    ranks, torch.tensor([3     , 4     ,2      ,0      ,1     ])), f"Calculated rank: {ranks}"
