from typing import List
from fvcore.common.param_scheduler import MultiStepParamScheduler
import numpy as np

from activedet.utils.math import find_upperbound

def difference(total):
    """Get the pairwise difference across each element.
    Example:
        total = [1,2,3,4,5]
        difference(total) # [1,1,1,1]
    Notes:
        Returned length = original length - 1
    """
    return [b - a for a, b in zip(total[:-1], total[1:])]

def get_milestones(checkpoints, max_iter):
    return [point / max_iter for point in checkpoints]

def get_active_multi_step_params(steps, checkpoints, max_iter, gamma):
    """Calculate The Multi-Step LR Schedular Parameter that fits the
    active learning process
    """
    steps = list(steps)
    checkpoints = list(checkpoints)
    total_checkpoints = [0] + list(checkpoints) + [max_iter]

    assert not any(
        [x in total_checkpoints for x in steps]
    ), f"Steps {steps} and checkpoints {checkpoints} must not have overlap"

    # Decrease the factor successively only within an active step.
    # For each active reset the multiplier into the original BASE_LR
    cur_upper_bound = 0
    factor = 1
    multiplier = []
    for step in steps:
        upper_bound = find_upperbound(step, total_checkpoints)

        if cur_upper_bound != upper_bound:
            cur_upper_bound = upper_bound
            factor = 1
        else:
            factor += 1

        multiplier.append(gamma ** factor)
    # It requires adding the checkpoints that defines new active step.
    # The reason is to reset the multiplier of LRScheduler.
    total_values = multiplier + [1] * len(checkpoints)
    total_steps = steps + checkpoints
    idx = np.argsort(total_steps)
    total_steps, total_values = (
        np.array(total_steps)[idx].tolist(),
        np.array(total_values)[idx].tolist(),
    )

    # Add initial multiplier
    total_values = [1] + total_values

    return total_values, total_steps


def get_multi_warmup_lengths(checkpoints, warmup_iters, max_iter):
    """Calculate the warmup length that is compatible with Active Learning Process
    """
    checkpoints = list(checkpoints)
    total_checkpoints = [0] + checkpoints + [max_iter]
    warmup_lengths = [
        min(warmup_iters / point, 1.0) for point in difference(total_checkpoints)
    ]

    return warmup_lengths

class ActiveMultiStepParamScheduler(MultiStepParamScheduler):
    # Improve precision to accommodate millions of iteration
    WHERE_EPSILON = 1e-7
    def __init__(self, checkpoints: List[int], steps: List[float], max_iter: int, gamma: float) -> None:
        total_values, total_steps = get_active_multi_step_params(steps,checkpoints,max_iter,gamma)
        super().__init__(values=total_values, milestones=total_steps,num_updates=max_iter)