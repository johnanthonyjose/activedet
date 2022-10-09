import torch
from torch.utils.data import TensorDataset

from activedet.data import ActiveDataset

def test_ActiveDataset_startn():
    dataset = TensorDataset(torch.arange(12))
    
    active_set = ActiveDataset(dataset, start_n=5)
    
    assert len(active_set) == 5


def test_ActiveDataset_split():
    dataset = TensorDataset(torch.arange(12))    
    active_set = ActiveDataset(dataset, start_n=5)

    train = active_set.dataset[:5][0]
    pool = active_set.pool[:7][0]

    combined = torch.cat(train,pool,dim=0).squeeze()
    v, ind = torch.sort(combined)

    assert v == torch.arange(12)


