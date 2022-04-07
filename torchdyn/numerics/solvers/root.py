# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Root finding solvers, line search utilities and root find API"""
import torch
from torch import einsum
from torch import norm
import numpy as np
from ..utils import RootLogger
from torch.autograd.functional import jacobian


class Broyden:
    """
        Template class for Broyden-type low-rank methods
    """
    type = 'Quasi-Newton'

    def __init__(self):
        pass

    def step(self, g, z0, J_inv, geval_old, alpha=1, **kwargs):
        dz = einsum('...o,...io->...i', geval_old, J_inv)
        z = z0 - alpha * dz
        geval = g(z)
        Δz, Δg = z - z0, geval - geval_old
        J_inv = self.update_jacobian(Δg, Δz, J_inv)
        return z, dz, geval, J_inv

    def update_jacobian(Δg, Δz, J_inv, **kwargs):
        raise NotImplementedError("")

class BroydenFull(Broyden):
    """
    """
    type = 'Quasi-Newton'

    def __init__(self):
        super().__init__()
        self.ε = 1e-6

    def update_jacobian(self, Δg, Δz, J_inv, **kwargs):
        num = Δz - einsum('...io, ...o -> ...i', J_inv, Δg)
        den = einsum('...i, ...io, ...o -> ...', Δz, J_inv, Δg) + self.ε
        prod = einsum('...i, ...io -> ...o', Δz, J_inv)
        ΔJ_inv = einsum('...i, ...o -> ...io', num / den[..., None], prod)
        J_inv = J_inv + ΔJ_inv
        return J_inv

class BroydenBad(Broyden):
    """Faster approximate Broyden method

    References
    ----------
    [Bro1965] Broyden, C G. *A class of methods for solving nonlinear
    simultaneous equations*. Mathematics of computation, 33 (1965),
    pp 577--593.

    [Kva1991] Kvaalen, E. *A faster Broyden method*. BIT Numerical
    Mathematics 31 (1991), pp 369--372.
    """
    type = 'Quasi-Newton'

    def __init__(self):
        super().__init__()
        self.Δg_tol = 1e-6

    def update_jacobian(self, Δg, Δz, J_inv, **kwargs):
        num = Δz - torch.einsum('...io, ...o -> ...i', J_inv, Δg)
        den = torch.sum(Δg**2, dim=1, keepdim=True) + self.Δg_tol
        ΔJ_inv = torch.einsum('...i, o... -> ...io', num, Δg.T) / den[..., None]
        J_inv = J_inv + ΔJ_inv
        return J_inv

class Newton():
    "Standard Newton iteration"
    type = 'Quasi-Newton'

    @staticmethod
    def step(g, geval_old, z0, J_inv, alpha=1, **kwargs):
        raise NotImplementedError

class Chord():
    "Standard newton iteration with precomputed J_inv"
    type = 'Quasi-Newton'

    @staticmethod
    def step(g, geval_old, z0, J_inv, alpha=1, **kwargs):
        raise NotImplementedError


##############################
### LINE SEARCH ALGORITHMS ###
##############################
def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


class LineSearcher(object):
    def __init__(self, g, g0, dz, z0):
        self.g, self.g0, self.dz, self.z0 = g, g0, dz, z0
        self.phi0 = _safe_norm(g0)**2

    def search(self, alpha0=1):
        raise NotImplementedError("")

    def phi(self, alpha):
        "Objective function for line search min_alpha phi(alpha)"
        return _safe_norm(self.g(self.z0 + alpha * self.dz))**2


class NaiveSearch(LineSearcher):
    def __init__(self, g, g0, dz, z0):
        super().__init__(g, g0, dz, z0)

    def search(self, alpha0=1, min_alpha=1e-6, mult_factor=0.1):
        alpha = alpha0
        phi_a0 = self.phi(alpha)
        while phi_a0 > self.phi0 and alpha > min_alpha:
            alpha = mult_factor*alpha
            phi_a0 = self.phi(alpha)
        return alpha, phi_a0


class LineSearchArmijo(LineSearcher):
    def __init__(self, g, g0, dz, z0):
        super().__init__(g, g0, dz, z0)

    def search(self, alpha0=1, c1=1e-4):
        """Minimize over alpha, the function ``phi(alpha)``.
            Uses the interpolation algorithm (Armijo backtracking) as suggested by
            Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
            alpha > 0 is assumed to be a descent direction.
            Returns
            -------
            alpha
            phi1
        """
        phi_a0 = self.phi(alpha0)
        derphi0 = -phi_a0 #TODO: check if this is correct <- derphi

        # if the objective function is <, return
        if phi_a0 < self.phi0 + c1*alpha0*derphi0: return alpha0, phi_a0

        # Otherwise, compute the minimizer of a quadratic interpolant
        alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - self.phi0 - derphi0 * alpha0)
        phi_a1 = self.phi(alpha1)

        # Loop with cubic interpolation until we find an alpha which
        # satisfies the first Wolfe condition (since we are backtracking, we will
        # assume that the value of alpha is not too small and satisfies the second
        # condition.
        while alpha1 > 1e-2:       # we are assuming alpha>0 is a descent direction
            factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
            a = alpha0**2 * (phi_a1 - self.phi0 - derphi0*alpha1) - \
                alpha1**2 * (phi_a0 - self.phi0 - derphi0*alpha0)
            a = a / factor
            b = -alpha0**3 * (phi_a1 - self.phi0 - derphi0*alpha1) + \
                alpha1**3 * (phi_a0 - self.phi0 - derphi0*alpha0)
            b = b / factor

            alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
            phi_a2 = self.phi(alpha2)

            if (phi_a2 <= self.phi0 + c1*alpha2*derphi0):
                return alpha2, phi_a2

            if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
                alpha2 = alpha1 / 2.0

            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi_a2

        # Failed to find a suitable step length
        return 1e-2, phi_a0


class LineSearchGriewank(LineSearcher):
    def __init__(self):
        super().__init__()

    def search(self):
        raise NotImplementedError


class LineSearchWolfe1(LineSearcher):
    def __init__(self):
        super().__init__()

    def search(self):
        raise NotImplementedError


class LineSearchWolfe2(LineSearcher):
    def __init__(self):
        super().__init__()

    def search(self):
        raise NotImplementedError

##############################
### TERMINATION CONDITIONS ###
##############################

class TerminationCondition(object):
    """
    Termination condition for an iteration. It is terminated if
    - (|F| < f_rtol*|F_0| && |F| < f_tol) && (|dx| < x_rtol*|x| && |dx| < x_tol)
    """

    def __init__(self, f_tol=1e-2, f_rtol=1e-1, x_tol=1e-2, x_rtol=1, iter=None):

        self.x_tol, self.x_rtol = x_tol, x_rtol
        self.f_tol, self.f_rtol = f_tol, f_rtol
        self.norm = norm
        self.iter = iter
        self.f0_norm = None
        self.iteration = 0

    def check(self, geval, z, dz):
        self.iteration += 1
        g_norm, z_norm, dz_norm = norm(geval, p=2, dim=1), norm(z, p=2, dim=1), norm(dz, p=2, dim=1)

        if self.f0_norm is None: self.f0_norm = g_norm

        cond1 = (g_norm <= self.f_tol).all() and (g_norm / self.f_rtol <= self.f0_norm).all()
        cond2 = (dz_norm <= self.x_tol).all() and (dz_norm / self.x_rtol <= z_norm).all()
        if cond1 and cond2: return 2

        return 0

#################
### ROOT FIND ###
#################

SEARCH_METHODS = {'naive': NaiveSearch, 'armijo': LineSearchArmijo, 'none': None}
ROOT_SOLVER_DICT = {'broyden_fast': BroydenBad(), 'broyden': BroydenFull(), 'newton': Newton, 'chord': Chord}

RETURN_CODES = {1: 'convergence',
                2: 'total condition'}

def batch_jacobian(func, x):
    "Batch Jacobian for 2D inputs of dimensions `bs, dims`"
    return torch.stack([jacobian(func, el) for el in x], 0)

def root_find(g, z, alpha=0.1, f_tol=1e-2, f_rtol=1e-1, x_tol=1e-2, x_rtol=1, maxiters=100,
              method='broyden', search_method='naive', verbose=True):
    assert method in ROOT_SOLVER_DICT, f'{method} not supported'
    assert search_method in SEARCH_METHODS, f'{search_method} not supported'

    tc = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol, x_tol=x_tol, x_rtol=x_rtol)
    logger = RootLogger()
    solver = ROOT_SOLVER_DICT[method]

    # first evaluation of g(z)
    geval = g(z)

    # initialize inverse jacobian J^-1g(z)
    J_inv = batch_jacobian(g, z).pinverse()

    iteration = 0
    while iteration <= maxiters:
        iteration += 1

        # solver step
        z, dz, geval, J_inv = solver.step(g=g, z0=z, J_inv=J_inv, geval_old=geval, alpha=alpha)

        # line search subroutines
        if SEARCH_METHODS[search_method] is not None:
            line_searcher = SEARCH_METHODS[search_method](g, geval, dz, z)
            alpha, phi = line_searcher.search()

        # logging
        if verbose and logger:
            logger.log({'geval': geval,
                        'z': z,
                        'dz': dz,
                        'iteration': iteration})

        # full termination check
        code = tc.check(geval, z, dz)
        if code > 0:
            if verbose and logger:
                logger.log({'termination_condition': RETURN_CODES[code]})
            break

    return z, logger