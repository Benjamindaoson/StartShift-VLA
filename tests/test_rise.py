import pytest

torch=pytest.importorskip("torch")
from torch import nn

from startshift.models.rise import RISEProjectorConfig, RISEStateProjector


def test_rise_linear_is_zero_init_equivalent_to_masked_base():
    torch.manual_seed(0)
    base=nn.Linear(32,12)
    projector=RISEStateProjector(base,RISEProjectorConfig(state_dim=8,mode="linear"))
    x=torch.randn(4,32)
    masked=torch.zeros_like(x)
    masked[:,:8]=x[:,:8]
    with torch.no_grad():
        expected=base(masked)
        actual=projector(x)
    torch.testing.assert_close(actual,expected)


def test_rise_adapter_starts_equivalent_and_can_learn_context():
    torch.manual_seed(0)
    base=nn.Linear(32,12)
    projector=RISEStateProjector(base,RISEProjectorConfig(state_dim=8,mode="adapter",bottleneck_dim=4))
    x=torch.randn(2,32)
    before=projector(x).detach()
    with torch.no_grad():
        projector.context[-1].weight.fill_(0.1)
    after=projector(x).detach()
    assert not torch.allclose(before,after)
