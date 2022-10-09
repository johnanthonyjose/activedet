from omegaconf import OmegaConf
from torchvision.transforms import transforms as T
from detectron2.config import LazyCall as L
from activedet.acquisition.heuristics import RandomHeuristic
from activedet.evaluation import MCEvaluator
from activedet.pool import trivial_context


active_learning = OmegaConf.create()
active_learning.start_n = 1000
active_learning.ndata_to_label = 1000
active_learning.epoch_per_step = 200
active_learning.drop_start = False
active_learning.pool = OmegaConf.create()
active_learning.pool.max_sample = 10000
active_learning.train_out_features = []
active_learning.start = OmegaConf.create()
active_learning.start.heuristic = None
active_learning.start.pool_evaluator = None
active_learning.start.model = None
active_learning.start.init_checkpoint = None
active_learning.start.max_sample = -1
active_learning.start_pool_transform= L(T.Compose)(
    transforms = [
        L(T.ToTensor)(),
        L(T.Normalize)(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ],
)

active_learning.heuristic = L(RandomHeuristic)(seed=0)

active_learning.pool_evaluator = L(MCEvaluator)(
        sample_size="${...dataloader.pool.num_repeats}",
        pool_batch_size="${...dataloader.pool.batch_size}",
        pool_context=trivial_context,
        # model and pool_builder is expected 
        # to be added before intantiate
    )