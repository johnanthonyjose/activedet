import math
from detectron2.config.config import configurable


def _checkpoint_from_config(cfg, *, start_iter=0):

    return {
        "start_n": cfg.ACTIVE_LEARNING.START_N,
        "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
        "epoch_per_step": cfg.ACTIVE_LEARNING.EPOCH_PER_STEP,
        "ndata_to_label": cfg.ACTIVE_LEARNING.NDATA_TO_LABEL,
        "max_iter": cfg.SOLVER.MAX_ITER,
        "drop_last": cfg.DATALOADER.DROP_LAST,
        "drop_start": cfg.ACTIVE_LEARNING.DROP_START,
        "start_iter": start_iter,
    }

@configurable(from_config=_checkpoint_from_config)
def calculate_checkpoints(
    start_n: int,
    ims_per_batch: int,
    epoch_per_step: int,
    ndata_to_label: int,
    max_iter: int,
    *,
    drop_last: bool = False,
    drop_start: bool = False,
    start_iter: int = 0
):
    checkpoints = []
    epochpoints = []

    # It assumes that it follows the same workflow in build_active_detection_train_loader
    dataset_size = start_n
    total_batch_size = ims_per_batch
    active_step = epoch_per_step
    iters_per_epoch = dataset_size / total_batch_size
    iters_per_epoch = math.floor(iters_per_epoch) if drop_last else math.ceil(iters_per_epoch)
    epochs = [
        (i + 1) * iters_per_epoch + start_iter for i in range(active_step)
    ]  # round down

    if drop_start:
        dataset_size -= start_n

    # TODO: What happens when the amount of pool is less than the amount to label?
    # The checkpoints must be able to check and change dynamically
    while epochs[-1] < max_iter:
        checkpoints.append(int(epochs[-1]))  # Round down to the nearest integer
        epochpoints += [int(e) for e in epochs]

        dataset_size += ndata_to_label
        iters_per_epoch = dataset_size / total_batch_size
        iters_per_epoch = math.floor(iters_per_epoch) if drop_last else math.ceil(iters_per_epoch)
        epochs = [
            (i + 1) * iters_per_epoch + epochs[-1] for i in range(active_step)
        ]
    return checkpoints, epochpoints
