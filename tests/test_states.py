import pytest
import torch
from activedet.utils.states import DetachMachinery, Detacher, MIAODTrainingPhase

@pytest.fixture
def features():
    return {
            "res2": torch.randn(2,2,requires_grad=True),
            "res3": torch.rand(1,3, requires_grad=True),
        }


def test_detacher_attach_id(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "attach"

    result = detacher(features)    
    assert id(result) == id(features), "It should be the same object"

def test_detacher_attach_keys(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "attach"

    result = detacher(features)    
    assert all(key1==key2 for key1,key2 in zip(result, features)), "All of their keys must be equal"

def test_detacher_attach_values(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "attach"

    result = detacher(features)    
    assert all(torch.equal(result[key1],features[key2]) for key1,key2 in zip(result,features)), "All of their values must be equal"

def test_detacher_attach_requires_grad(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "attach"

    result = detacher(features)    
    assert all(result[key].requires_grad == True for key in result), "It must be able to maintain requires_grad since it's attached"

def test_detacher_detach_keys(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "detach"
    result = detacher(features)    

    assert all(key1==key2 for key1,key2 in zip(result, features)), "All of their keys must be equal"

def test_detacher_detach_values(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "detach"

    result = detacher(features)    
    assert all(torch.equal(result[key1],features[key2]) for key1,key2 in zip(result,features)), "All of their values must be equal"

def test_detacher_detach_requires_grad(features):
    points = [[],[]]
    names = ["melt","freeze"]
    detacher = Detacher(conditions=points,names=names)
    # Will be added by Machine
    detacher.state = "detach"

    result = detacher(features)    
    assert all(result[key].requires_grad == False for key in result), "It must make requires_grad into False since it's detached"

def test_DetachMachinery_states():
    expected = ["detach","attach"]
    assert all(state in DetachMachinery.states for state in expected)
    assert len(DetachMachinery.states) == 2

def test_DetachMachinery_initial_state():
    machinery = DetachMachinery([100,200],[150])
    assert machinery.state == "attach"

def test_DetachMachinery_state_dict():
    machinery = DetachMachinery([100,200],[150])
    state_dict = machinery.state_dict()
    assert "state" in state_dict.keys()
    assert "last_iter" in state_dict.keys()
    assert state_dict["state"] == "attach"
    assert state_dict["last_iter"] == 0

def test_DetachMachinery_before2_detach_state():
    train_iter = 98
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    assert machinery.state == "attach"

def test_DetachMachinery_before2_detach_requires_grad(features):
    train_iter = 98
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    result = machinery(features)
    assert all(result[key].requires_grad == True for key in result), "It must be able to maintain requires_grad since it's attached"

def test_DetachMachinery_before_detach_state():
    train_iter = 99
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    machinery.step()
    assert machinery.state == "detach"

def test_DetachMachinery_before_detach_requires_grad(features):
    train_iter = 99
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    result = machinery(features)
    assert all(result[key].requires_grad == True for key in result), "It must be able to maintain requires_grad since it's attached"

def test_DetachMachinery_on_detach_state():
    train_iter = 100
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    machinery.step()
    assert machinery.state == "detach"

def test_DetachMachinery_on_detach_requires_grad(features):
    train_iter = 100
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    result = machinery(features)
    assert all(result[key].requires_grad == False for key in result), "It must make requires_grad into False since it's detached"

def test_DetachMachinery_after_detach_state(features):
    train_iter = 100
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    machinery.step()
    machinery.step()
    assert machinery.state == "detach"

def test_DetachMachinery_after_detach_requires_grad(features):
    train_iter = 100
    machinery = DetachMachinery([100,200],[150], last_iter=train_iter-1)
    machinery.step()
    machinery.step()
    result = machinery(features)
    assert all(result[key].requires_grad == False for key in result), "It must make requires_grad into False since it's detached"

def test_DetachMachinery_load_state_dict():
    train_iter = 148
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"detach","last_iter": train_iter-1})
    assert machinery.state == "detach"
    assert machinery.last_iter == train_iter - 1

def test_DetachMachinery_before2_attach_state():
    train_iter = 148
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"detach","last_iter": train_iter-1})
    machinery.step()
    assert machinery.state == "detach"

def test_DetachMachinery_before2_attach_requires_grad(features):
    train_iter = 148
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"detach","last_iter": train_iter})
    result = machinery(features)
    assert all(result[key].requires_grad == False for key in result), "It must make requires_grad into False since it's detached"

def test_DetachMachinery_before_attach_state():
    train_iter = 149
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"detach","last_iter": train_iter})
    machinery.step()
    assert machinery.state == "attach"

def test_DetachMachinery_before_attach_requires_grad(features):
    train_iter = 149
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"detach","last_iter": train_iter})
    result = machinery(features)
    assert all(result[key].requires_grad == False for key in result), "It must make requires_grad into False since it's detached"

def test_DetachMachinery_on_attach_state():
    train_iter = 150
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"attach","last_iter": train_iter})
    machinery.step()
    assert machinery.state == "attach"

def test_DetachMachinery_on_attach_requires_grad(features):
    train_iter = 150
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"attach","last_iter": train_iter})
    result = machinery(features)
    assert all(result[key].requires_grad == True for key in result), "It must be able to maintain requires_grad since it's attached"

def test_DetachMachinery_after_attach_state():
    train_iter = 151
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"attach","last_iter": train_iter})
    machinery.step()
    assert machinery.state == "attach"

def test_DetachMachinery_after_attach_requires_grad(features):
    train_iter = 151
    machinery = DetachMachinery([100,200],[150])
    machinery.load_state_dict({"state":"attach","last_iter": train_iter})
    result = machinery(features)
    assert all(result[key].requires_grad == True for key in result), "It must be able to maintain requires_grad since it's attached"

def test_MIAODTrainingPhase_states():
    expected = ["labeled","min_uncertainty","max_uncertainty"]
    assert all(state in MIAODTrainingPhase.states for state in expected)
    assert len(MIAODTrainingPhase.states) == 3

def test_MIAODTrainingPhase_initial_state():
    machinery = MIAODTrainingPhase([100,200],[150],[180])
    assert machinery.state == "labeled"

def test_MIAODTrainingPhase_state_dict():
    machinery = MIAODTrainingPhase([100,200],[150],[180])
    state_dict = machinery.state_dict()
    assert "state" in state_dict.keys()
    assert "last_iter" in state_dict.keys()
    assert state_dict["state"] == "labeled"
    assert state_dict["last_iter"] == 0

def test_MIAODTrainingPhase_before2_min_state():
    train_iter = 148
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.step()
    assert machinery.state == "labeled"

def test_MIAODTrainingPhase_before_min_state():
    train_iter = 149
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.step()
    assert machinery.state == "min_uncertainty"

def test_MIAODTrainingPhase_on_min_state():
    train_iter = 150
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"min_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "min_uncertainty"

def test_MIAODTrainingPhase_after_min_state():
    train_iter = 151
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"min_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "min_uncertainty"

def test_MIAODTrainingPhase_after2_min_state():
    train_iter = 152
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"min_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "min_uncertainty"

def test_MIAODTrainingPhase_before2_max_state():
    train_iter = 178
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"min_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "min_uncertainty"

def test_MIAODTrainingPhase_before_max_state():
    train_iter = 179
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"min_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "max_uncertainty"

def test_MIAODTrainingPhase_on_max_state():
    train_iter = 180
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"max_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "max_uncertainty"

def test_MIAODTrainingPhase_after_max_state():
    train_iter = 181
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"max_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "max_uncertainty"

def test_MIAODTrainingPhase_after2_max_state():
    train_iter = 182
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"max_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "max_uncertainty"

def test_MIAODTrainingPhase_before2_label_state():
    train_iter = 198
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"max_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "max_uncertainty"

def test_MIAODTrainingPhase_before_label_state():
    train_iter = 199
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"max_uncertainty", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "labeled"

def test_MIAODTrainingPhase_on_label_state():
    train_iter = 200
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"labeled", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "labeled"

def test_MIAODTrainingPhase_after_label_state():
    train_iter = 201
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"labeled", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "labeled"

def test_MIAODTrainingPhase_after2_label_state():
    train_iter = 202
    machinery = MIAODTrainingPhase([100,200],[150],[180],last_iter=train_iter-1)
    machinery.load_state_dict({"state":"labeled", "last_iter":train_iter})
    machinery.step()
    assert machinery.state == "labeled"