from typing import List, Any
import torch

def find_upperbound(target: int, bounds: List[int]) -> int:
    """Looks for the higher nearest neighbour  of x on the given list
    Args:
        target : reference value
        bounds: range of possible neighbors
    Returns:
        higher neighbor of x
    Example:
        Case 1 - Trivial
            target = 3
            bounds = [-1, 2, 10, 20]
            find_upperbound(target, bounds) # Output: 10
        Case 2 - When bounds is not sorted
            target = 3
            bounds = [20, 10, -1, 2]
            find_upperbound(target, bounds) # Output: 10
        Case 3 - When target and bounds have identical values
            target = 5
            bounds = [-3, 3, 5, 8, 9]
            find_upperbound(target, bounds) # Output: 8
        Case 4 - When target is out of bounds
            target = 100
            bounds = [-3, 3, 5, 8, 9]
            find_upperbound(target, bounds) # Output: None
    """
    neighbors = sorted(bounds)
    matches = [x for x in neighbors if x > target]
    if matches:
        return matches[0]
    
    return None

def normalize(x, dim=1, eps=1e-8):
    x_n = x.norm(dim=dim)[:, None]
    return x / torch.max(x_n, eps * torch.ones_like(x_n))

def cosine_distance(a, b, eps=1e-8):
    """Calculates the cosine similarity according to the ff equations:
        cos(a,b) = dot(a, b) / (norm(a) * norm(b) )
    It includes eps for numerical stability
    The main difference from pytorch is that it is able to calculate
    the cosine similarity even if a and b have different first dimension
    Args:
        a : input 1 with dim of x1 x N
        b : input 2 with dim of x2 x N
    Returns:
        a matrix with dim of x1 x x2
    """
    a_norm = normalize(a, eps=eps)
    b_norm = normalize(b, eps=eps)
    cos_matrix = torch.mm(a_norm, b_norm.transpose(0, 1))
    return cos_matrix

class State:
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class BoundsTransition:
    def __init__(self, bounds: List[int]):
        self.bounds = bounds

    def condition(self, iteration):
        return iteration in self.bounds
class StateMachine:
    def __init__(self, states: List[State], initial: State) -> None:
        self.states = states
        self.current_state = initial
        detach_bounds = BoundsTransition(bounds=[1000,2000,3000])
        re_attach_bounds = BoundsTransition(bounds=[500,1500,2500])

        self.transition = [
            {"source": "detach", "dest": "attach","condition": re_attach_bounds},
            {"source": "attach", "dest": "detach", "condition": detach_bounds},
        ]

    @property
    def state(self):
        return self.current_state
        
    def next(self):
        pass