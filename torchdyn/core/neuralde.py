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

from typing import Callable, Union, List

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
    def __init__(self, vector_field, solver:Union[str, nn.Module]='tsit5', order:int=1, atol:float=1e-3, rtol:float=1e-3, sensitivity='autograd',
                 solver_adjoint:Union[str, nn.Module, None] = None, atol_adjoint:float=1e-4, rtol_adjoint:float=1e-4, 
                 interpolator:Union[str, Callable, None]=None, integral_loss:Union[Callable, None]=None, seminorm:bool=False,
                 return_t_eval:bool=True):
        """Generic Neural Ordinary Differential Equation. 

        Args:
            vector_field ([type]): [description]
            solver (Union[str, nn.Module]): [description]
            order (int, optional): [description]. Defaults to 1.
            atol (float, optional): [description]. Defaults to 1e-4.
            rtol (float, optional): [description]. Defaults to 1e-4.
            sensitivity (str, optional): [description]. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): [description]. Defaults to None.
            atol_adjoint (float, optional): [description]. Defaults to 1e-6.
            rtol_adjoint (float, optional): [description]. Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): [description]. Defaults to None.
            seminorm (bool, optional): [description]. Defaults to False.
        """
        super().__init__(vector_field=standardize_vf_call_signature(vector_field, order, defunc_wrap=True), order=order, sensitivity=sensitivity,
                         solver=solver, atol=atol, rtol=rtol, atol_adjoint=atol_adjoint, rtol_adjoint=rtol_adjoint, 
                         seminorm=seminorm, interpolator=interpolator, integral_loss=integral_loss)
        self.u, self.controlled, self.t_span = None, False, None # data-control conditioning
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
            if hasattr(module, 'u'):
                self.controlled = True
                module.u = x[:, excess_dims:].detach()
        return x, t_span

    def forward(self, x:Tensor, t_span:Tensor=None):
        x, t_span = self._prep_integration(x, t_span)
        t_eval, sol =  super().forward(x, t_span)
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
    """General Neural SDE class
    :param drift_func: function parametrizing the drift.
    :type drift_func: nn.Module
    :param diffusion_func: function parametrizing the diffusion.
    :type diffusion_func: nn.Module
    :param settings: specifies parameters of the Neural DE.
    :type settings: dict
    """
    def __init__(self, drift_func, 
                       diffusion_func, 
                       noise_type ='diagonal',
                       sde_type = 'ito',
                       order=1,
                       sensitivity='autograd',
                       s_span=torch.linspace(0, 1, 2),
                       solver='srk',
                       atol=1e-4,
                       rtol=1e-4,
                       ds = 1e-3,
                       intloss=None):
        super().__init__(func=SDEFunc(f=drift_func, g=diffusion_func, order=order), order=order, sensitivity=sensitivity, s_span=s_span, solver=solver,
                                      atol=atol, rtol=rtol)
        if order != 1: raise NotImplementedError
        self.defunc.noise_type, self.defunc.sde_type = noise_type, sde_type
        self.adaptive = False
        self.intloss = intloss
        self.u, self.controlled = None, False  # datasets-control
        self.ds = ds

    def _prep_sdeint(self, x:torch.Tensor):
        self.s_span = self.s_span.to(x)
        # datasets-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
        # TO DO: merge the named_modules loop for perf
        excess_dims = 0
        for _, module in self.defunc.named_modules():
            if hasattr(module, 'u'):
                self.controlled = True
                module.u = x[:, excess_dims:].detach()

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
    def __init__(self):
        super().__init__()