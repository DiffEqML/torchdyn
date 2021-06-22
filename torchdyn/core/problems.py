from inspect import getfullargspec
import torch
from torch.autograd import Function
from torch import Tensor
import torch.nn as nn
from typing import Union, List

from torchdyn.core.defunc import DEFuncBase
from torchdyn.numerics.sensitivity import _gather_odefunc_adjoint, _gather_odefunc_interp_adjoint
from torchdyn.numerics.odeint import odeint, str_to_solver


class ODEProblem(nn.Module):
    def __init__(self, vector_field, solver:Union[str, nn.Module], order:int=1, atol:float=1e-4, rtol:float=1e-4, sensitivity='autograd',
                 solver_adjoint:Union[str, nn.Module, None] = None, atol_adjoint:float=1e-6, rtol_adjoint:float=1e-6, seminorm:bool=False):
        """An ODE Problem coupling a given vector field with solver and sensitivity algorithm to compute gradients w.r.t different quantities.

        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): [description]
            order (int, optional): [description]. Defaults to 1.
            atol (float, optional): [description]. Defaults to 1e-4.
            rtol (float, optional): [description]. Defaults to 1e-4.
            sensitivity (str, optional): [description]. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): [description]. Defaults to None.
            atol_adjoint (float, optional): [description]. Defaults to 1e-6.
            rtol_adjoint (float, optional): [description]. Defaults to 1e-6.
            seminorm (bool, optional): Indicates whether the a seminorm should be used for error estimation during adjoint backsolves. Defaults to False.
        
        """
        super().__init__()
        # instantiate solver at initialization
        if type(solver) == str: 
            solver = str_to_solver(solver)
        if solver_adjoint is None:
            solver_adjoint = solver
        else: solver_adjoint = str_to_solver(solver_adjoint)

        self.solver, self.atol, self.rtol = solver, atol, rtol
        self.solver_adjoint, self.atol_adjoint, self.rtol_adjoint = solver_adjoint, atol_adjoint, rtol_adjoint

        # wrap vector field if `t, x` is not the call signature
        if issubclass(type(vector_field), nn.Module):
            if 't' not in getfullargspec(vector_field.forward).args:
                print("Your vector field callable (nn.Module) should have both time `t` and state `x` as arguments, "
                    "we've wrapped it for you.")
                vector_field = DEFuncBase(vector_field)
        else: 
            # argspec for lambda functions needs to be done on the function itself
            if 't' not in getfullargspec(vector_field).args:
                print("Your vector field callable (lambda) should have both time `t` and state `x` as arguments, "
                    "we've wrapped it for you.")
                vector_field = DEFuncBase(vector_field, has_time_arg=False)   
            else: vector_field = DEFuncBase(vector_field, has_time_arg=True) 

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
            self.autograd_function = _gather_odefunc_adjoint(self.vf, self.vf_params, solver, atol, rtol,
                                                            solver_adjoint, atol_adjoint, rtol_adjoint).apply
        elif self.sensalg == 'interpolated_adjoint':
            self.autograd_function = _gather_odefunc_interp_adjoint(self.vf, self.vf_params, solver, atol, rtol,
                                                                    solver_adjoint, atol_adjoint, rtol_adjoint).apply

    def odeint(self, x:Tensor, t_span:Tensor):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == 'autograd':
            return odeint(self.vf, x, t_span, self.solver, self.atol, self.rtol)
        else:
            return self.autograd_function(self.vf_params, x, t_span)

    def forward(self, x:Tensor, t_span:Tensor):
        "For safety redirects to intented method `odeint`"
        return self.odeint(x, t_span)


class MultipleShootingProblem(nn.Module):
    def __init__(self, solver:str, vector_field, sensalg='autograd'):
        """[summary]

        Args:
            solver (str): [description]
            vector_field ([type]): [description]
            sensalg (str, optional): [description]. Defaults to 'autograd'.

        Returns:
            [type]: [description]
        """
        super().__init__()
        #
        self.solver
        self.sensalg, self.vf, self.solver = sensalg, vf, solver

        #TODO: this fails when vf does not have parameters
        if len(tuple(vf.parameters())) > 0:
            self.vector_field = torch.cat([p.contiguous().flatten() for p in vector_field.parameters()])
        else:
            self.vf_params = nn.Parameter(torch.zeros(1))

        if 't' not in getfullargspec(vector_field.forward).args:
            self.vf = DEFuncBase(vector_field)

        if self.sensalg == 'adjoint':  # alias .apply as direct call to preserve consistency of call signature
            self.odefunc = _gather_odefunc_adjoint(self.vf, self.vf_params, self.solver, atol, rtol).apply
        elif self.sensalg == 'interpolated_adjoint':
            self.odefunc = _gather_odefunc_interp_adjoint(self.vf, self.vf_params, self.solver, atol, rtol).apply
        else:
            def odefunc(vf_params, x0, t_span, t_eval=[]):
                return odeint(self.vf, x=x0, t_span=t_span,
                              t_eval=t_eval, solver=self.solver, atol=atol, rtol=rtol)

            self.odefunc = odefunc

    def forward(self, x0, t_span, atol=1e-4, rtol=1e-4, t_eval=[]):
        x0, t_span = prep_input(x0, t_span)
        t_eval, sol = self.odefunc(self.vf_params, x0, t_span, t_eval)
        return t_eval, sol
        
