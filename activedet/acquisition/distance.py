import torch

def maximum(distances: torch.Tensor) -> int:
    """Takes distances value of a certain array
    Returns the index of the maximum value
    Args:
        distances: a 1D tensor of distance values of each data point
    Returns:
        index of the maximum value
    """
    val, indices = torch.sort(distances)
    return indices[-1]

def minimum(distances: torch.Tensor) -> int:
    """Takes distances value of a certain array
    Returns the index of the minum value
    Args:
        distances: a 1D tensor of distance values of each data point
    Returns:
        index of the maximum value
    """
    val, indices = torch.sort(distances)
    return indices[0]
class Percentile:
    """Calculates nth Percentile
    Args:
        nth (float): It indicates on which percentile we are interested to collect
            The value must be between 0-1
    """
    def __init__(self, nth: float):
        self.nth = nth
        pass
    def __call__(self, distances: torch.Tensor):
        """Takes distances value of a certain array
        Returns the index of the nth percentile value
        Args:
            distances: a 1D tensor of distance values of each data point
        Returns:
            index of the nth percentile
        """
        val, indices = torch.sort(distances)
        return indices[int(len(indices) * self.nth)]

dist_score_function = {
    "max": maximum,
    "min": minimum,
    "median": Percentile(0.50),
    "quartile3": Percentile(0.75),
    "quartile1": Percentile(0.25),
    "percentile90": Percentile(0.90),
    "percentile95": Percentile(0.95),
    "percentile99": Percentile(0.99),
    }


def batched_all_pairs_squared_l2_dist(
    a: torch.FloatTensor, b: torch.FloatTensor
) -> torch.FloatTensor:
    """For each batch, return the squared L2 distance between each pair of vectors
    Let A and B be tensors of shape NxM_AxD and NxM_BxD, each containing N*M_A
    and N*M_B vectors of dimension D grouped in N batches of size M_A and M_B.
    For each batch, for each vector of A and each vector of B, return the sum
    of the squares of the differences of their components.
    """
    num_chunks, num_a, dim = a.shape
    num_b = b.shape[1]
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[2]
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)
    # Calculate res_i,k = sum_j((a_i,j - b_k,j)^2) for each i and k as
    # sum_j(a_i,j^2) - 2 sum_j(a_i,j b_k,j) + sum_j(b_k,j^2), by using a matrix
    # multiplication for the ab part, adding the b^2 as part of the baddbmm call
    # and the a^2 afterwards.
    res = torch.baddbmm(b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2).add_(
        a_squared.unsqueeze(-1)
    )
    assert res.shape == (num_chunks, num_a, num_b)

    return res


def batched_all_pairs_l2_dist(
    a: torch.FloatTensor, b: torch.FloatTensor
) -> torch.FloatTensor:
    squared_res = batched_all_pairs_squared_l2_dist(a, b)
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res