from torchvision import models
from activedet.utils.intermediate_layer import get_child_module


def test_get_child_module_identical1():
    model = models.resnet50(pretrained=False)
    child = get_child_module(model,".")

    assert id(model) == id(child)

def test_get_child_module_identical2():
    model = models.resnet50(pretrained=False)
    child = get_child_module(model,"")

    assert id(model) == id(child)

def test_get_child_module_identical3():
    model = models.resnet50(pretrained=False)
    child = get_child_module(model,None)

    assert id(model) == id(child)


def test_get_child_module_child():
    model = models.resnet50(pretrained=False)
    child = get_child_module(model,"layer1")

    assert id(model.layer1) == id(child)


def test_get_child_module_grandchild():
    model = models.resnet50(pretrained=False)
    child = get_child_module(model,"layer1.0")

    assert id(model.layer1[0]) == id(child)
