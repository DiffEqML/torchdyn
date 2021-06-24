import pytest
from torchdyn.numerics.root import *
from .conftest import *


@pytest.mark.parametrize('func', [quad, cubic, ackley])
@pytest.mark.parametrize('method', ['broyden'])
def test_root_iteration(method, func):
    z0 = 0.1 * torch.randn(4, 2)

    root, log = root_find(func, z0,
                         alpha=1, search_method='armijo',
                         f_tol=1e-3, f_rtol=1e-3, x_tol=1e-2, x_rtol=1e-2,
                         maxiters=1000, method=method)
    assert (func(root) <= 1e-2).all()
