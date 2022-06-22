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

from typing import Callable, Union, List, Iterable, Generator

from torchdyn.core.problems import MultipleShootingProblem, ODEProblem, SDEProblem
from torchdyn.numerics import odeint
from torchdyn.core.defunc import DEFunc, DEFuncBase, SDEFunc
from torchdyn.core.utils import standardize_vf_call_signature

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torchsde

import warnings


class NeuralODE(ODEProblem, pl.LightningModule):
    def __init__(self, vector_field:Union[Callable, nn.Module], solver:Union[str, nn.Module]='tsit5', order:int=1, 
                atol:float=1e-3, rtol:float=1e-3, sensitivity='autograd', solver_adjoint:Union[str, nn.Module, None] = None, 
                atol_adjoint:float=1e-4, rtol_adjoint:float=1e-4, interpolator:Union[str, Callable, None]=None, \
                integral_loss:Union[Callable, None]=None, seminorm:bool=False, return_t_eval:bool=True, optimizable_params:Union[Iterable, Generator]=()):
        """Generic Neural Ordinary Differential Equation.

        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): 
            order (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): Defaults to None.
            seminorm (bool, optional): Whether to use seminorms for adaptive stepping in backsolve adjoints. Defaults to False.
            return_t_eval (bool): Whether to return (t_eval, sol) or only sol. Useful for chaining NeuralODEs in `nn.Sequential`.
            optimizable_parameters (Union[Iterable, Generator]): parameters to calculate sensitivies for. Defaults to ().
        Notes:
            In `torchdyn`-style, forward calls to a Neural ODE return both a tensor `t_eval` of time points at which the solution is evaluated
            as well as the solution itself. This behavior can be controlled by setting `return_t_eval` to False. Calling `trajectory` also returns
            the solution only. 

            The Neural ODE class automates certain delicate steps that must be done depending on the solver and model used. 
            The `prep_odeint` method carries out such steps. Neural ODEs wrap `ODEProblem`.
        """
        super().__init__(vector_field=standardize_vf_call_signature(vector_field, order, defunc_wrap=True), order=order, sensitivity=sensitivity,
                         solver=solver, atol=atol, rtol=rtol, solver_adjoint=solver_adjoint, atol_adjoint=atol_adjoint, rtol_adjoint=rtol_adjoint, 
                         seminorm=seminorm, interpolator=interpolator, integral_loss=integral_loss, optimizable_params=optimizable_params)
        self._control, self.controlled, self.t_span = None, False, None # data-control conditioning
        self.return_t_eval = return_t_eval
        if integral_loss is not None: self.vf.integral_loss = integral_loss
        self.vf.sensitivity = sensitivity

    def _prep_integration(self, x:Tensor, t_span:Tensor) -> Tensor:
        "Performs generic checks before integration. Assigns data control inputs and augments state for CNFs"

        # assign a basic value to `t_span` for `forward` calls that do no explicitly pass an integration interval
        if t_span is None and self.t_span is None: t_span = torch.linspace(0, 1, 2)
        elif t_span is None: t_span = self.t_span

        # loss dimension detection routine; for CNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (not self.integral_loss is None) and self.sensitivity == 'autograd':
            excess_dims += 1

        # handle aux. operations required for some jacobian trace CNF estimators e.g Hutchinson's
        # as well as datasets-control set to DataControl module
        for _, module in self.vf.named_modules():
            if hasattr(module, 'trace_estimator'):
                if module.noise_dist is not None: module.noise = module.noise_dist.sample((x.shape[0],))
                excess_dims += 1

            # data-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
            if hasattr(module, '_control'):
                self.controlled = True
                module._control = x[:, excess_dims:].detach()
        return x, t_span

    def forward(self, x:Tensor, t_span:Tensor=None, save_at:Iterable=(), args={}):
        x, t_span = self._prep_integration(x, t_span)
        t_eval, sol =  super().forward(x, t_span, save_at, args)
        if self.return_t_eval: return t_eval, sol
        else: return sol

    def trajectory(self, x:torch.Tensor, t_span:Tensor):
        x, t_span = self._prep_integration(x, t_span)
        _, sol = odeint(self.vf, x, t_span, solver=self.solver, atol=self.atol, rtol=self.rtol)
        return sol

    def __repr__(self):
        npar = sum([p.numel() for p in self.vf.parameters()])
        return f"Neural ODE:\n\t- order: {self.order}\
        \n\t- solver: {self.solver}\n\t- adjoint solver: {self.solver_adjoint}\
        \n\t- tolerances: relative {self.rtol} absolute {self.atol}\
        \n\t- adjoint tolerances: relative {self.rtol_adjoint} absolute {self.atol_adjoint}\
        \n\t- num_parameters: {npar}\
        \n\t- NFE: {self.vf.nfe}"


class NeuralSDE(SDEProblem, pl.LightningModule):
    def __init__(self, drift_func, diffusion_func, noise_type ='diagonal', sde_type = 'ito', order=1,
                 sensitivity='autograd', s_span=torch.linspace(0, 1, 2), solver='srk',
                 atol=1e-4, rtol=1e-4, ds = 1e-3, intloss=None):
        """Generic Neural Stochastic Differential Equation. Follows the same design of the `NeuralODE` class.

        Args:
            drift_func ([type]): drift function
            diffusion_func ([type]): diffusion function
            noise_type (str, optional): Defaults to 'diagonal'.
            sde_type (str, optional): Defaults to 'ito'.
            order (int, optional): Defaults to 1.
            sensitivity (str, optional): Defaults to 'autograd'.
            s_span ([type], optional): Defaults to torch.linspace(0, 1, 2).
            solver (str, optional): Defaults to 'srk'.
            atol ([type], optional): Defaults to 1e-4.
            rtol ([type], optional): Defaults to 1e-4.
            ds ([type], optional): Defaults to 1e-3.
            intloss ([type], optional): Defaults to None.

        Raises:
            NotImplementedError: higher-order Neural SDEs are not yet implemented, raised by setting `order` to >1.

        Notes:
            The current implementation is rougher around the edges compared to `NeuralODE`, and is not guaranteed to have the same features.
        """
        super().__init__(func=SDEFunc(f=drift_func, g=diffusion_func, order=order), order=order, sensitivity=sensitivity, s_span=s_span, solver=solver,
                                      atol=atol, rtol=rtol)
        if order != 1: raise NotImplementedError
        self.defunc.noise_type, self.defunc.sde_type = noise_type, sde_type
        self.adaptive = False
        self.intloss = intloss
        self._control, self.controlled = None, False  # datasets-control
        self.ds = ds

    def _prep_sdeint(self, x:torch.Tensor):
        self.s_span = self.s_span.to(x)
        # datasets-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
        excess_dims = 0
        for _, module in self.defunc.named_modules():
            if hasattr(module, '_control'):
                self.controlled = True
                module._control = x[:, excess_dims:].detach()

        return x

    def forward(self, x:torch.Tensor):
        x = self._prep_sdeint(x)
        switcher = {
            'autograd': self._autograd,
            'adjoint': self._adjoint,
        }
        sdeint = switcher.get(self.sensitivity)
        out = sdeint(x)
        return out
    
    def trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        x = self._prep_sdeint(x)
        sol = torchsde.sdeint(self.defunc, x, s_span, rtol=self.rtol, atol=self.atol, 
                              method=self.solver, dt=self.ds)
        return sol
    
    def backward_trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        raise NotImplementedError
        
    def _autograd(self, x):
        self.defunc.intloss, self.defunc.sensitivity = self.intloss, self.sensitivity
        return torchsde.sdeint(self.defunc, x, self.s_span, rtol=self.rtol, atol=self.atol,
                                   adaptive=self.adaptive, method=self.solver, dt=self.ds)[-1]
    
    def _adjoint(self, x):
        out = torchsde.sdeint_adjoint(self.defunc, x, self.s_span, rtol=self.rtol, atol=self.atol,
                     adaptive=self.adaptive, method=self.solver, dt=self.ds)[-1]
        return out


class MultipleShootingLayer(MultipleShootingProblem, pl.LightningModule):
    def __init__(self, vector_field:Callable, solver:str, sensitivity:str='autograd',
                 maxiter:int=4, fine_steps:int=4, solver_adjoint:Union[str, nn.Module, None] = None, atol_adjoint:float=1e-6, 
                 rtol_adjoint:float=1e-6, seminorm:bool=False, integral_loss:Union[Callable, None]=None):
        """Multiple Shooting Layer as defined in https://arxiv.org/abs/2106.03885. 
        
        Uses parallel-in-time ODE solvers to solve an ODE parametrized by neural network `vector_field`. 

        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): parallel-in-time solver, ['zero', 'direct']
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            maxiter (int): number of iterations of the root finding routine defined to parallel solve the ODE.
            fine_steps (int): number of fine-solver steps to perform in each subinterval of the parallel solution.
            solver_adjoint (Union[str, nn.Module, None], optional): Standard sequential ODE solver for the adjoint system. 
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): Currently not implemented
            seminorm (bool, optional): Whether to use seminorms for adaptive stepping in backsolve adjoints. Defaults to False.
        Notes:
            The number of shooting parameters (first dimension in `B0`) is implicitly defined by passing `t_span` during forward calls.
            For example, a `t_span=torch.linspace(0, 1, 10)` will define 9 intervals and 10 shooting parameters.
            
            For the moment only a thin wrapper around `MultipleShootingProblem`. At this level will be convenience routines for special
            initializations of shooting parameters `B0`, as well as usual convenience checks for integral losses.
        """
        super().__init__(vector_field, solver, sensitivity, maxiter, fine_steps, solver_adjoint, atol_adjoint,
                         rtol_adjoint, seminorm, integral_loss)
                    

