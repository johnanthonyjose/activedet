from typing import Dict, List, Any
from functools import partial
import torch
from transitions import Machine


class Weather:
    """Helper Class that stores break points (similar to checkpoints)
    It dynamically adds new attributes and methods at instantiated object.

    Args:
        conditions(List[List[int]]): Each list in the list indicates where
            a specific condition-name pair will be true
        names (List[str]): Describes the name of the new method to be
            included in the instantiated object. Note: It must have the
            same length as conditions
    Example:
        conditions = [[1,5],[3,8]]
        name = ["melt","freeze"]
        weather = Weather(conditions,name)
        # The instanstiated object will have methods named as melt and freeze
        weather.melt(5) # True
        weather.melt(6) # False
        weather.freeze(8) # True
        weather.frezee(9) # False


    """

    def __init__(self, conditions: List[List[int]], names: List[str]) -> None:
        assert len(conditions) == len(names), "Length of name and conditions must match"

        for n, breaks in zip(names, conditions):
            setattr(self, n, partial(Weather.conditioner, checks=breaks))

    @staticmethod
    def conditioner(i: int, checks: List[int]) -> bool:
        """Describes on the behavior of each instanstiated method
        Args:
            i (int): indicates the current training iteration
            checks (List[int]): indicates on which break points will this method be true
        Returns:
            True if i is in the breakpoints. Otherwise, False.
        """
        return i in checks


class Detacher(Weather):
    """Responsible for Detach functionality for tensors.
    Inherits Weather so that it provides the functionality to specify
    which iterations will it start to detach tensors or not.
    Attributes:
        meltpoints(List[int]) : indicates on which iteration will it start
            to detach an input tensor
        freezepoints(List[int]): indicates on which iteration will it recover the
            regular behavior(i.e. attached) on an input tensor
    Notes:
        - detach in tensor means that the tensor (to be detached) will not be part of
        the underlying computation graph.
        - Additionally, all further computation using the detached tensor is also not part of the
        computation graph.
        - In Pytorch, we know that it is not already included when the grad of the said tensor
        is already False. (please see unit test)
    """

    def detach(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: features.detach() for name, features in input.items()}

    def __call__(self, features):
        assert self.state
        return self.detach(features) if self.state == "detach" else features


class DetachMachinery:
    """Trigger-Based Detacher using Finite State Machine Data Structure
    Attributes:
        detacher (Detacher): responsible for detaching and re-attaching tensor according to breakpoints
        machine (Machine): state machine responsible for indicating whether to attach or detach
        last_iter (int): Last iteration on where machinery was used. Similar to optimizer
        states (List[str]): List of possible states monitored by the state machine.
    Args:
        detach_points(List[int]) : indicates on which iteration will it start
            to detach an input tensor
        re_attach_points(List[int]): indicates on which iteration will it recover the
            regular behavior(i.e. attached) on an input tensor
        last_iter (int): `same as attributes`
    """

    states: List[str] = ["attach", "detach"]

    def __init__(
        self, detach_points: List[int], re_attach_points: List[int], last_iter=-1
    ):
        breakpoints = [detach_points, re_attach_points]
        condition_name = ["melt", "freeze"]
        self.detacher = Detacher(conditions=breakpoints, names=condition_name)
        self.machine = Machine(
            model=self.detacher, states=DetachMachinery.states, initial="attach"
        )
        self.machine.add_transition("next", "attach", "detach", conditions="melt")
        self.machine.add_transition("next", "detach", "attach", conditions="freeze")
        self.last_iter = last_iter
        self.step()

    def __call__(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Detaches the input tensor depending on the current state
        Args:
            input : dictionaries of input tensor that follows detectron2 resnet output format
        Returns:
            if the current state is "detach", then all tensors in input are detached from computation graph
            if the current state is "attach", then it is equivalent to input
        """
        return self.detacher(input)

    def step(self) -> None:
        """Calculates the current state of the state machine.
        Responsible for identifying whether the next iteration will change the state or not
        """
        # Assumes that the next increment is only added by 1
        self.last_iter += 1
        self.detacher.next(self.last_iter)

    @property
    def state(self) -> str:
        return self.detacher.state

    def state_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "last_iter": self.last_iter}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.machine.set_state(state_dict["state"])
        self.last_iter = state_dict["last_iter"]


class MIAODTrainingPhase:
    """Monitors the proper training phases as indicated in the paper
    `Multiple Instance Active Learning for Object Detection`
    Attributes:
        train_watcher (Weather): responsible for monitoring the proper training phase according to breakpoints
        machine (Machine): state machine responsible for transitioning machine states
        last_iter (int): Last iteration on where machinery was used. Similar to optimizer
        states (List[str]): List of possible states monitored by the state machine.
    Args:
        checkpoints(List[int]) : indicates on which iteration does training restarts into its original phase
        minimize_poins(List[int]): indicates on which iteration will it start to transition into 
            Minimizing Instance Uncertainty
        maximize_poins(List[int]): indicates on which iteration will it start to transition into 
            Maximizing Instance Uncertainty
        last_iter (int): `same as attributes`
    """

    states = ["labeled", "min_uncertainty", "max_uncertainty"]

    def __init__(
        self,
        checkpoints: List[int],
        minimize_poins: List[int],
        maximize_points: List[int],
        last_iter=-1,
    ):
        conditions = [minimize_poins, maximize_points, checkpoints]
        names = ["melt", "freeze", "rain"]
        self.train_watcher = Weather(conditions=conditions, names=names)
        self.machine = Machine(
            model=self.train_watcher,
            states=MIAODTrainingPhase.states,
            initial="labeled",
        )
        self.machine.add_transition(
            "next", "labeled", "min_uncertainty", conditions="melt"
        )
        self.machine.add_transition(
            "next", "max_uncertainty", "min_uncertainty", conditions="melt"
        )
        self.machine.add_transition(
            "next", "min_uncertainty", "max_uncertainty", conditions="freeze"
        )
        self.machine.add_transition(
            "next", "labeled", "max_uncertainty", conditions="freeze"
        )
        self.machine.add_transition(
            "next", "max_uncertainty", "labeled", conditions="rain"
        )
        self.machine.add_transition(
            "next", "min_uncertainty", "labeled", conditions="rain"
        )

        self.last_iter = last_iter
        self.step()

    def step(self) -> None:
        """Calculates the current state of the state machine.
        Responsible for identifying whether the next iteration will change the state or not
        """
        # Assumes that the next increment is only added by 1
        self.last_iter += 1
        self.train_watcher.next(self.last_iter)

    @property
    def state(self) -> str:
        return self.train_watcher.state

    def state_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "last_iter": self.last_iter}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.machine.set_state(state_dict["state"])
        self.last_iter = state_dict["last_iter"]
