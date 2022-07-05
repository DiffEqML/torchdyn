import torch
import torch.nn as nn

# TODO: work around circular imports
# multiple shooting solvers are "composite": they use
# several "base" solvers and this are further down in the dependency chain
# likely solution: place composite solver templates in a different file

# from torchdyn.numerics.solvers.ode import str_to_solver, str_to_ms_solver


class DiffEqSolver(nn.Module):
    def __init__(
            self, 
            order, 
            stepping_class:str="fixed", 
            min_factor:float=0.2, 
            max_factor:float=10, 
            safety:float=0.9
        ):

        super(DiffEqSolver, self).__init__()
        self.order = order
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None
        self.stepping_class = stepping_class

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        device = x.device
        if self.tableau is not None:
            c, a, bsol, berr = self.tableau
            self.tableau = c.to(x), [a.to(x) for a in a], bsol.to(x), berr.to(x)
        t_span = t_span.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)
        return x, t_span

    def step(self, f, x, t, dt, k1=None, args=None):
        raise NotImplementedError("Stepping rule not implemented for the solver")


class BaseExplicit(DiffEqSolver):
    def __init__(self, *args, **kwargs):
        """Base template for an explicit differential equation solver
        """
        super(BaseExplicit, DiffEqSolver).__init__(*args, **kwargs)
        assert self.stepping_class in ["fixed", "adaptive"]



class BaseImplicit(DiffEqSolver):
    def __init__(self, *args, **kwargs):
        """Base template for an implicit differential equation solver
        """
        super(BaseImplicit, DiffEqSolver).__init__(*args, **kwargs)
        assert self.stepping_class in ["fixed", "adaptive"]

    @staticmethod
    def _residual(f, x, t, dt, x_sol):
        raise NotImplementedError


class MultipleShootingDiffeqSolver(nn.Module):
    def __init__(self, coarse_method, fine_method):
        super(MultipleShootingDiffeqSolver, self).__init__()
        # if type(coarse_method) == str: self.coarse_method = str_to_solver(coarse_method)
        # if type(fine_method) == str: self.fine_method = str_to_solver(fine_method)

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        x, t_span = self.coarse_method.sync_device_dtype(x, t_span)
        x, t_span = self.fine_method.sync_device_dtype(x, t_span)
        return x, t_span

    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        raise NotImplementedError