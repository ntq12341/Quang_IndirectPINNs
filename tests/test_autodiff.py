import torch
from src.autodiff import outward_normal_derivative, spatial_derivatives


def test_spatial_and_normal_derivatives():
    for side, expected_normal in ((0, -1.0), (1, 4.0)):
        x = torch.tensor([[float(side)]], dtype=torch.float64, requires_grad=True)
        value = x**3 + x
        first, second = spatial_derivatives(value, x)
        assert torch.allclose(first, 3 * x**2 + 1)
        assert torch.allclose(second, 6 * x)
        assert torch.allclose(outward_normal_derivative(value, x, side), torch.tensor([[expected_normal]], dtype=torch.float64))
