import torch
import torch.nn.functional as F

def cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.cross_entropy(input, target, reduction=reduction, **kwargs)




class Linear(torch.nn.Linear):
    """ A wrapper on :class:`torch.nn.Linear` to support extra features
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, norm = None, activation = None) -> None:
        """ Add extra keywords argument support 
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        self.norm = norm
        self.activation = activation
        super().__init__(in_features=in_features,out_features=out_features,bias=bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.linear(input, self.weight, self.bias)

        if self.norm is not None:
            x = self.norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x
