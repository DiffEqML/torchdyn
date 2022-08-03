import torch
import abc
from torchdyn.numerics.solvers.templates import DiffEqSolver
from torchsde._brownian import BaseBrownian
from torchdyn.core.defunc import SDEFunc

def _check_types(sde, solver_sde_type, solver_noise_type):
    if sde.sde_type != solver_sde_type:
        raise ValueError(f"SDE's type : {sde.sde_type} and solver type : {solver_sde_type} is different!")
    if sde.noise_type not in solver_noise_type:
        raise ValueError(f"Solver only supports noise types of {solver_noise_type} but noise type of {sde.noise_type} is given by SDE")

# todo : Should I make additional SDEqSolver?? so many duplicates here
class EulerMaruyama(DiffEqSolver):
    def __init__(self, sde:SDEFunc, bm:BaseBrownian, dtype=torch.float32):
        super().__init__(order=1)
        self.sde = sde
        self.bm = bm
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.solver_sde_type = 'ito'
        self.solver_noise_type = ['general','diagonal','scalar','additive']
        _check_types(self.sde, self.solver_sde_type, self.solver_noise_type)

    def step(self, x, t, dt):

        next_t = t+dt
        x_sol = x + self.sde.f(t,x)*dt + self.sde.g(t,x) * self.bm(t,next_t)

        return None, x_sol, None

class EulerHeun(DiffEqSolver):
    def __init__(self, sde:SDEFunc, bm:BaseBrownian, dtype=torch.float32):
        super().__init__(order=1)
        self.sde = sde
        self.bm = bm
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.solver_sde_type = 'stratonovich'
        self.solver_noise_type = ['general','diagonal','scalar','additive']
        _check_types(self.sde, self.solver_sde_type, self.solver_noise_type)

    def step(self, x, t, dt):

        next_t = t + dt
        f = self.sde.f(t,x)
        g = self.sde.g(t,x) 
        x_prime = x + g * self.bm(t,next_t)

        x_sol = x + f * dt  + (g + self.sde.g(next_t, x_prime))*0.5 * self.bm(t,next_t)

        return None, x_sol, None

class MilsteinBase(DiffEqSolver):
    # todo : Derivative-free Milstein Method only for now. Will add Derivative version as well in the future.
    def __init__(self, sde:SDEFunc, bm:BaseBrownian, dtype=torch.float32):
        super().__init__(order=1)
        self.sde = sde
        self.bm = bm
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.solver_noise_type = ['general','diagonal','scalar','additive']

    @abc.abstractmethod
    def v_term(self,bm_, dt):
        raise NotImplementedError

    def step(self, x, t, dt):
        next_t = t + dt
        bm_ = self.bm(t, next_t)
        v = self.v_term(bm_, dt)
        f = self.sde.f(t,x)
        g = self.sde.g(t,x) 
        sqrt_dt = dt.sqrt()
        x_prime = x + f * dt + g * sqrt_dt
        g_x_prime = self.sde.g(t, x_prime)
        gdg = g_x_prime - g

        x_sol = x + f * dt + g *bm_ + (gdg * 1/(2*sqrt_dt))*v

        return None, x_sol, None

class MilesteinIto(MilsteinBase):
    def __init__(self, sde: SDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(sde, bm, dtype)
        self.solver_sde_type = 'ito'

    def v_term(self,bm_, dt):
        return bm_**2 - dt

class MilesteinStratonovich(MilsteinBase):
    def __init__(self, sde: SDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(sde, bm, dtype)
        self.solver_sde_type = 'stratonovich'

    def v_term(self,bm_, dt):
        return bm_**2 

class Midpoint(DiffEqSolver):
    def __init__(self, bm:BaseBrownian, dtype=torch.float32):
        super().__init__(order=2)
        pass

SDE_SOLVER_DICT = {
    'euler': EulerMaruyama, 'eulerHeun':EulerHeun, 'midpoint': Midpoint, 'milstein_ito': MilesteinIto, 
    'milstein_stratonovich':MilesteinStratonovich}

def sde_str_to_solver(solver_name, sde:SDEFunc, bm:BaseBrownian, dtype=torch.float32):
    "Transforms string specifying desired solver into an instance of the Solver class."
    solver = SDE_SOLVER_DICT[solver_name]
    return solver(sde, bm, dtype)