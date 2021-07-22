import torch
from torch.autograd import Function
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, List

from torchdyn.core.defunc import DEFuncBase
from torchdyn.numerics.sensitivity import _gather_odefunc_adjoint, _gather_odefunc_interp_adjoint
from torchdyn.numerics.odeint import odeint, odeint_mshooting, str_to_solver
from torchdyn.core.utils import standardize_vf_call_signature


class ODEProblem(nn.Module):
    def __init__(self, vector_field:Union[Callable, nn.Module], solver:Union[str, nn.Module], interpolator:Union[str, Callable, None]=None, order:int=1, 
                atol:float=1e-4, rtol:float=1e-4, sensitivity:str='autograd', solver_adjoint:Union[str, nn.Module, None] = None, atol_adjoint:float=1e-6, 
                rtol_adjoint:float=1e-6, seminorm:bool=False, integral_loss:Union[Callable, None]=None):
        """An ODE Problem coupling a given vector field with solver and sensitivity algorithm to compute gradients w.r.t different quantities.

        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): [description]
            order (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            seminorm (bool, optional): Indicates whether the a seminorm should be used for error estimation during adjoint backsolves. Defaults to False.
            integral_loss (Union[Callable, None]): Integral loss to optimize for. Defaults to None.

        Notes:
            Integral losses can be passed as generic function or `nn.Modules`. 
        """
        super().__init__()
        # instantiate solver at initialization
        if type(solver) == str: solver = str_to_solver(solver)
        if solver_adjoint is None:
            solver_adjoint = solver
        else: solver_adjoint = str_to_solver(solver_adjoint)

        self.solver, self.interpolator, self.atol, self.rtol = solver, interpolator, atol, rtol
        self.solver_adjoint, self.atol_adjoint, self.rtol_adjoint = solver_adjoint, atol_adjoint, rtol_adjoint
        self.sensitivity, self.integral_loss = sensitivity, integral_loss
        
        # wrap vector field if `t, x` is not the call signature
        vector_field = standardize_vf_call_signature(vector_field)

        self.vf, self.order, self.sensalg = vector_field, order, sensitivity
        if len(tuple(self.vf.parameters())) > 0:
            self.vf_params = torch.cat([p.contiguous().flatten() for p in self.vf.parameters()])
        else:
            print("Your vector field does not have `nn.Parameters` to optimize.")
            dummy_parameter = self.vf_params = nn.Parameter(torch.zeros(1))
            self.vf.register_parameter('dummy_parameter', dummy_parameter)

        # instantiates an underlying autograd.Function that overrides the backward pass with the intended version
        # sensitivity algorithm
        if self.sensalg == 'adjoint':  # alias .apply as direct call to preserve consistency of call signature
            self.autograd_function = _gather_odefunc_adjoint(self.vf, self.vf_params, solver, atol, rtol, interpolator, 
                                                            solver_adjoint, atol_adjoint, rtol_adjoint, integral_loss,
                                                            problem_type='standard').apply
        elif self.sensalg == 'interpolated_adjoint':
            self.autograd_function = _gather_odefunc_interp_adjoint(self.vf, self.vf_params, solver, atol, rtol, interpolator, 
                                                                    solver_adjoint, atol_adjoint, rtol_adjoint, integral_loss,
                                                                    problem_type='standard').apply

    def odeint(self, x:Tensor, t_span:Tensor):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == 'autograd':
            return odeint(self.vf, x, t_span, self.solver, self.atol, self.rtol, interpolator=self.interpolator)
        else:
            return self.autograd_function(self.vf_params, x, t_span)

    def forward(self, x:Tensor, t_span:Tensor):
        "For safety redirects to intended method `odeint`"
        return self.odeint(x, t_span)


class MultipleShootingProblem(ODEProblem):
    def __init__(self, vector_field:Callable, solver:str, sensitivity:str='autograd',
                 maxiter:int=4, fine_steps:int=4, solver_adjoint:Union[str, nn.Module, None] = None, atol_adjoint:float=1e-6, 
                 rtol_adjoint:float=1e-6, seminorm:bool=False, integral_loss:Union[Callable, None]=None):
        """An ODE problem solved with parallel-in-time methods.

        Args:
            vector_field (Callable):  the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                    In the second case, the Callable is automatically wrapped for consistency
            solver (str): parallel-in-time solver.
            sensitivity (str, optional): [description]. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): [description]. Defaults to None.
            atol_adjoint (float, optional): [description]. Defaults to 1e-6.
            rtol_adjoint (float, optional): [description]. Defaults to 1e-6.
            seminorm (bool, optional): [description]. Defaults to False.
            integral_loss (Union[Callable, None], optional): [description]. Defaults to None.
        """
        super().__init__(vector_field=vector_field, solver=None, interpolator=None, order=1, 
                sensitivity=sensitivity, solver_adjoint=solver_adjoint, atol_adjoint=atol_adjoint, 
                rtol_adjoint=rtol_adjoint, seminorm=seminorm, integral_loss=integral_loss)
        self.parallel_solver = solver
        self.fine_steps, self.maxiter = fine_steps, maxiter

        if self.sensalg == 'adjoint':  # alias .apply as direct call to preserve consistency of call signature
            self.autograd_function = _gather_odefunc_adjoint(self.vf, self.vf_params, solver, 0, 0, None, 
                                                            solver_adjoint, atol_adjoint, rtol_adjoint, integral_loss,
                                                            'multiple_shooting', fine_steps, maxiter).apply
        elif self.sensalg == 'interpolated_adjoint':
            self.autograd_function = _gather_odefunc_interp_adjoint(self.vf, self.vf_params, solver, 0, 0, None, 
                                                                    solver_adjoint, atol_adjoint, rtol_adjoint, integral_loss,
                                                                    'multiple_shooting', fine_steps, maxiter).apply

    def odeint(self, x:Tensor, t_span:Tensor, B0:Tensor=None):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == 'autograd':
            return odeint_mshooting(self.vf, x, t_span, self.parallel_solver, B0, self.fine_steps, self.maxiter)
        else:
            return self.autograd_function(self.vf_params, x, t_span, B0)

    def forward(self, x:Tensor, t_span:Tensor, B0:Tensor=None):
        "For safety redirects to intended method `odeint`"
        return self.odeint(x, t_span, B0)
        

class SDEProblem(nn.Module):
    def __init__(self):
        "Extension of `ODEProblem` to SDE"
        super().__init__()
        raise NotImplementedError("Hoperfully soon...")
