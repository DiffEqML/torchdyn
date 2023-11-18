import pytest
from torch import nn
import torch
import torchsde
import numpy as np
from torchdyn.numerics import sdeint
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize("solver", ["euler", "milstein_ito"])
def test_geo_brownian_ito(solver):
    torch.manual_seed(0)
    np.random.seed(0)

    t0, t1 = 0, 1
    size = (1, 1)
    device = "cpu"

    alpha = torch.sigmoid(torch.normal(mean=0.0, std=1.0, size=size)).to(device)
    beta = torch.sigmoid(torch.normal(mean=0.0, std=1.0, size=size)).to(device)
    x0 = torch.normal(mean=0.0, std=1.1, size=size).to(device)
    t_size = 1000
    ts = torch.linspace(t0, t1, t_size).to(device)

    bm = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=size, device=device, levy_area_approximation="space-time"
    )

    def get_bm_queries(bm, ts):
        bm_increments = torch.stack(
            [bm(t0, t1) for t0, t1 in zip(ts[:-1], ts[1:])], dim=0
        )
        bm_queries = torch.cat(
            (torch.zeros(1, 1, 1).to(device), torch.cumsum(bm_increments, dim=0))
        )
        return bm_queries

    class SDE(nn.Module):
        def __init__(self, alpha, beta):
            super().__init__()
            self.alpha = nn.Parameter(alpha, requires_grad=True)
            self.beta = nn.Parameter(beta, requires_grad=True)
            self.noise_type = "diagonal"
            self.sde_type = "ito"

        def f(self, t, x):
            return self.alpha * x

        def g(self, t, x):
            return self.beta * x

    sde = SDE(alpha, beta).to(device)

    with torch.no_grad():
        _, xs_torchdyn = sdeint(sde, x0, ts, solver=solver, bm=bm)

    bm_queries = get_bm_queries(bm, ts)
    xs_true = x0.cpu() * np.exp(
        (alpha.cpu() - 0.5 * beta.cpu() ** 2) * ts.cpu()
        + beta.cpu() * bm_queries[:, 0, 0].cpu()
    )

    assert_almost_equal(xs_true[0][-1], xs_torchdyn[-1], decimal=2)


@pytest.mark.parametrize("solver", ["eulerHeun", "milstein_stratonovich"])
def test_geo_brownian_stratonovich(solver):
    torch.manual_seed(0)
    np.random.seed(0)

    t0, t1 = 0, 1
    size = (1, 1)
    device = "cpu"

    alpha = torch.sigmoid(torch.normal(mean=0.0, std=1.0, size=size)).to(device)
    beta = torch.sigmoid(torch.normal(mean=0.0, std=1.0, size=size)).to(device)
    x0 = torch.normal(mean=0.0, std=1.1, size=size).to(device)
    t_size = 1000
    ts = torch.linspace(t0, t1, t_size).to(device)

    bm = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=size, device=device, levy_area_approximation="space-time"
    )

    def get_bm_queries(bm, ts):
        bm_increments = torch.stack(
            [bm(t0, t1) for t0, t1 in zip(ts[:-1], ts[1:])], dim=0
        )
        bm_queries = torch.cat(
            (torch.zeros(1, 1, 1).to(device), torch.cumsum(bm_increments, dim=0))
        )
        return bm_queries

    class SDE(nn.Module):
        def __init__(self, alpha, beta):
            super().__init__()
            self.alpha = nn.Parameter(alpha, requires_grad=True)
            self.beta = nn.Parameter(beta, requires_grad=True)
            self.noise_type = "diagonal"
            self.sde_type = "stratonovich"

        def f(self, t, x):
            return self.alpha * x

        def g(self, t, x):
            return self.beta * x

    sde = SDE(alpha, beta).to(device)

    with torch.no_grad():
        _, xs_torchdyn = sdeint(sde, x0, ts, solver=solver, bm=bm)

    bm_queries = get_bm_queries(bm, ts)
    xs_true = x0.cpu() * np.exp(
        (alpha.cpu() - 0.5 * beta.cpu() ** 2) * ts.cpu()
        + beta.cpu() * bm_queries[:, 0, 0].cpu()
    )

    assert_almost_equal(xs_true[0][-1] - xs_torchdyn[-1], 1, decimal=0)

