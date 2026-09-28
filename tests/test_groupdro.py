import pytest

torch=pytest.importorskip("torch")
from startshift.training.robust import GroupDRO


def test_groupdro_upweights_hard_group():
    dro=GroupDRO(num_groups=2,eta=0.2)
    losses=torch.tensor([0.1,0.2,2.0,2.2])
    groups=torch.tensor([0,0,1,1])
    loss,stats=dro(losses,groups)
    assert loss.item()>0
    assert stats["group_weights"][1]>stats["group_weights"][0]
