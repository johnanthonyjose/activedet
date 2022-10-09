from typing import List
from functools import reduce

import torch.nn as nn

def get_child_module(model: nn.Module, node: str) -> nn.Module:
    """Converts a string into the expected nn.Module
        Args:
            model: the model to be trained
            node: string representation where the hook would be registered on
        Returns:
            nn.Module inside the model as dictated by node
    """
    if node == "." or node == "" or node is None:
        return model
        
    module_list = node.split(".")
    module = model
    if module_list:
        module = reduce(getattr, [module, *module_list])

    return module

class LayerHook:
    """Taps into the intermediate layer of a CNN Model

    Attributes:
        inputs (List[Any]) : List of input sent in forward
        outputs (List[Any]) : List of outputs after calling forward
        hookList (List[nn.Module]): List of modules have registered with forward_hook
    
    Examples:
        hooker = LayerHook()
        model = models.resnet50()
        hooker.register(model, ["layer1.conv1","fc"])

        hooker.outputs[0] #Output of the intermediate layer, layer1.conv1
        hooker.outputs[1] #Output of the intermediate layer, fc

        # Clear it to save VRAM space in GPU. Avoids out of memory error
        hooker.clear()
    """

    def __init__(self):
        self.outputs = []
        self.inputs = []
        self.hookList = []

    def hook(self, module, input, output):
        """Function required by register_forward_hook
        """
        self.inputs.append(input)
        self.outputs.append(output)

    def register(self, model: nn.Module, nodes: List[str]) -> None:
        """Register nodes with a forward hook
        Args:
            model : the model to be trained
            nodes : expected modules to be registered with forward_hook in string
        Examples:
            Acceptable string representation for nodes list
            Each dot (.) represents a nested operation.
            X.Y = there's a module named X, inside of which there's Y module    
            X = the module named X in the model.
        """
        for node in nodes:
            module = get_child_module(model, node)
            hook = module.register_forward_hook(self.hook)
            self.hookList.append(hook)

    def remove(self) -> None:
        """Removes the hook registrations.
        Useful for re-inserting new hooks
        """
        [hook.remove() for hook in self.hookList]
    
    def clear(self) -> None:
        """Clears the content inside the hook buffers
        """
        self.inputs.clear()
        self.outputs.clear()