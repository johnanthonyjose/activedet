from detectron2.config import CfgNode as CN

def add_active_learning_config(cfg) -> None:
    """Add config on Active Learning
    Adding config here is stateful. Meaning,
    it will update the input arg. No need for output
    Args:
        cfg = config that has pre-loaded detectron2 defaults
    Returns:
        None
    Example:
        cfg = setup(args)
        add_active_learning_config(cfg)
    """
    cfg.MODEL.AUX = CN()
    cfg.MODEL.AUX.LOSS_PREDICTION = CN()

    # Auxilliary Module to be added on top of the exising model. If None, there's no Aux module
    cfg.MODEL.AUX.NAME = None
    # A module that describes on how aux module calculates loss function
    cfg.MODEL.AUX.PREDICTOR = "LossPrediction"
    # Describes which out_features in the model are to be used as target of Aux
    cfg.MODEL.AUX.OUT_FEATURES = ["loss_cls","loss_box_reg"]
    # Describes which in_features in the model are to be used as features of Aux
    cfg.MODEL.AUX.IN_FEATURES = ["backbone"]
    # Describes where Loss prediction will be tapped on. Each element describes one loss prediciton module
    cfg.MODEL.AUX.LOSS_PREDICTION.IN_FEATURES = ["p2","p3","p4","p5"] 
    # Describes the Pool Resolution wherein the features will be resized upon
    cfg.MODEL.AUX.LOSS_PREDICTION.POOLER_RESOLUTIONS = [1,1,1,1]
    # Describes the output feature width of each prediction module. It must have the same length as IN_FEATURES
    cfg.MODEL.AUX.LOSS_PREDICTION.OUTPUT_WIDTH = [128,128,128,128]
    # Allowable margin to be consider the pair of losses as higher
    cfg.MODEL.AUX.LOSS_PREDICTION.MARGIN = 1
    # Weighting penalty hyperparameter for learning loss relative to other loss
    cfg.MODEL.AUX.LOSS_PREDICTION.LAMBDA = 1
    # Provides the ability for ROI Head to pred loss in batch-style format rather than scalar output
    cfg.MODEL.ROI_HEADS.IS_REDUCE = "True"
    # Whether to add dropout on the RetinaNetHead or not
    cfg.MODEL.RETINANET.USE_DROPOUT = False

    cfg.MODEL.RESNETS.DROPOUT_RATE = 0.0
    cfg.DATASETS.INIT_SAMPLING_METHOD = CN()
    cfg.DATASETS.INIT_SAMPLING_METHOD.NAME = "Random"    # {"Random", "BalancedLookAhead"}
    cfg.DATASETS.INIT_SAMPLING_METHOD.HEURISTIC = None    # {None, 'largest_area'}
    cfg.ACTIVE_LEARNING = CN()
    cfg.ACTIVE_LEARNING.START_N = 40 # amount of initial samples on the first step
    cfg.ACTIVE_LEARNING.DROP_START = False #Drop START_N data after its initial training. We use its weights across each step.
    cfg.ACTIVE_LEARNING.NDATA_TO_LABEL = 20 # number of samples to label per active learningstep
    cfg.ACTIVE_LEARNING.POOL = CN()
    cfg.ACTIVE_LEARNING.POOL.BATCH_SIZE=16 # Amount of samples being inferred for each iter
    cfg.ACTIVE_LEARNING.POOL.ESTIMATOR = "MCDropout" # The type of bayesian estimator used during pool inference
    cfg.ACTIVE_LEARNING.POOL.EVALUATOR = "MCEvaluator" # {"MCEvaluator", "GNEvaluator"}
    cfg.ACTIVE_LEARNING.POOL.SAMPLING_METHOD = "MonteCarlo" # Describes the Sampling Method that complements the estimator
    cfg.ACTIVE_LEARNING.POOL.MC_SIZE=10 # Amount of Monte Carlo Samples
    cfg.ACTIVE_LEARNING.POOL.GRID_SIZE=16 # Reducer size after prediction.
    cfg.ACTIVE_LEARNING.POOL.MAX_SAMPLE = -1 # max subsample of pool to take into considertaion during acquistion. -1 means all
    cfg.ACTIVE_LEARNING.POOL.SD_LEVELS = [0,8,16,24,32,40,48] # Standard Deviation of the Gaussian Noise Added to the noisy images, Pixel Value assumes [0, 255]
                                                                # should start from 0, as this would include denote the original image, used in conjuction with GNEvaluator
    cfg.ACTIVE_LEARNING.EPOCH_PER_STEP = 3 # no. of epochs before updating the dataset
    cfg.ACTIVE_LEARNING.CLS_MERGE_MODE = "max" # {'max', 'mean', 'sum'}
    cfg.ACTIVE_LEARNING.HEURISTIC = CN()
    cfg.ACTIVE_LEARNING.HEURISTIC.NAME = "Random" # {Random, SemSegHeuristic, ClassificationEntropy}
    cfg.ACTIVE_LEARNING.HEURISTIC.SCORE_THRESH = 0.0 # threshold to filter low detection scores during heuristics function. 
                                                   # (SCORE_THRESH = 0 if no thresholding is required)
    cfg.ACTIVE_LEARNING.HEURISTIC.TOP_N = 0 # top_n detections (with highest classification scores) to consider during heuristics function.
                                             # (TOP_N = 0 to consider all detections per image)
    cfg.ACTIVE_LEARNING.CORESET = CN()
    cfg.ACTIVE_LEARNING.CORESET.IN_FEATURE = 'p6' # feature level of FPN to be used for the feature vector of the image, {'p2','p3','p4','p5','p6'}
    cfg.ACTIVE_LEARNING.CORESET.FEATURE_HEIGHT = 15
    cfg.ACTIVE_LEARNING.CORESET.FEATURE_WIDTH = 15
    cfg.ACTIVE_LEARNING.HEURISTIC.IMPUTE_VALUE = 1000.0 # Value to be imputed when Instances has no prediction

    # Enables Pytorch AMP Training
    cfg.SOLVER.AMP.ENABLED = True
    # Used for `poly` learning rate schedule.
    # cfg.SOLVER.LR_SCHEDULER_NAME = "Constant"
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0

    # Enables evaluations per active step
    cfg.TEST.WITH_EVALUATION = True

    # Enables calculations of validation loss
    cfg.TEST.WITH_VALIDATION = False

    # Enables / Disable Dropping of residual batch in train dataloader
    cfg.DATALOADER.DROP_LAST = False
