import torch
from src.models import IndirectPINN


def test_indirect_hard_constraints():
    torch.manual_seed(7)
    model = IndirectPINN(final_time=1.0).double()
    t = torch.linspace(0, 1, 17, dtype=torch.float64).reshape(-1, 1)
    for side in (0, 1):
        x = torch.full_like(t, float(side))
        assert torch.equal(model.state(x, t), model.control(t, side))
        assert torch.equal(model.adjoint(x, t), torch.zeros_like(t))
    x = torch.linspace(0, 1, 17, dtype=torch.float64).reshape(-1, 1)
    terminal = torch.ones_like(x)
    assert torch.equal(model.adjoint(x, terminal), torch.zeros_like(x))

