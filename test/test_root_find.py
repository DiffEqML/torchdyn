import pytest
from torchdyn.numerics.solvers.root import *
from .conftest import *


@pytest.mark.parametrize('func', [quad, cubic, ackley])
@pytest.mark.parametrize('method', ['broyden', 'broyden_fast'])
@pytest.mark.parametrize('search_method', ['armijo']) #
def test_root_iteration(func, method, search_method):
    z0 = 0.01 * torch.randn(1000, 2)
    root, log = root_find(func, z0,
                         alpha=.1, search_method=search_method,
                         f_tol=1e-3, f_rtol=1e-3, x_tol=1e-2, x_rtol=1e-2,
                         maxiters=500, method=method)
    assert (func(root) <= 1e-2).all()
