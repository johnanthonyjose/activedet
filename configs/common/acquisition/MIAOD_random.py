from .YooKwon_random import active_learning

active_learning.start_n = 16551 // 20
active_learning.ndata_to_label = 16551 // 40
active_learning.epoch_per_step = 26
active_learning.pool.max_sample = -1