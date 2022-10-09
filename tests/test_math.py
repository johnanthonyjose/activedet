from activedet.utils.math import find_upperbound


def test_find_upperbound_trivial():
    target = 3
    bounds = [-1, 2, 10, 20]
    assert find_upperbound(target,bounds) == 10

def test_find_upperbound_unsorted():
    target = 3
    bounds = [20, 10, -1, 2]
    assert find_upperbound(target,bounds) == 10

def test_find_upperbound_identical():
    target = 5
    bounds = [-3, 3, 5, 8, 9]    
    assert find_upperbound(target,bounds) == 8

def test_find_upperbound_outbound():
    target = 100
    bounds = [-3, 3, 5, 8, 9]    
    assert find_upperbound(target,bounds) is None