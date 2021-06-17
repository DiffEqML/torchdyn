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

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchdiffeq
import torchsde
from torchdyn.numerics.adjoint import Adjoint

from torchdyn.core.defunc import DEFunc, SDEFunc
from torchdyn.core.utils import SCIPY_SOLVERS
import warnings


class NeuralDETemplate(pl.LightningModule):
    """General Neural DE template"""
    def __init__(self, func,
                       order=1,
                       sensitivity='autograd',
                       s_span=torch.linspace(0, 1, 2),
                       solver='rk4',
                       atol=1e-4,
                       rtol=1e-4,
                       intloss=None):
        super().__init__()
        self.defunc = func
        self.order = order
        self.sensitivity, self.s_span, self.solver = sensitivity, s_span, solver
        self.nfe = self.defunc.nfe
        self.rtol, self.atol = rtol, atol
        self.intloss = intloss
        self.u, self.controlled = None, False # datasets-control

    def reset(self):
        self.nfe, self.defunc.nfe = 0, 0

    @property
    def nfe(self):
        return self.defunc.nfe

    @nfe.setter
    def nfe(self, val):
        self.defunc.nfe = val

    def __repr__(self):
        npar = sum([p.numel() for p in self.defunc.parameters()])
        return f"Neural DE:\n\t- order: {self.order}\
        \n\t- solver: {self.solver}\n\t- integration interval: {self.s_span[0]} to {self.s_span[-1]}\
        \n\t- num_checkpoints: {len(self.s_span)}\
        \n\t- tolerances: relative {self.rtol} absolute {self.atol}\
        \n\t- num_parameters: {npar}\
        \n\t- NFE: {self.nfe}\n\
        \nIntegral loss: {self.intloss}\n\
        \nDEFunc:\n {self.defunc}"


class NeuralODE(NeuralDETemplate):
    """General Neural ODE class

    :param func: function parametrizing the vector field.
    :type func: nn.Module
    :param settings: specifies parameters of the Neural DE.
    :type settings: dict
    """
    def __init__(self, func:nn.Module,
                       order=1,
                       sensitivity='autograd',
                       s_span=torch.linspace(0, 1, 2),
                       solver='rk4',
                       atol=1e-4,
                       rtol=1e-4,
                       intloss=None):
        super().__init__(func=DEFunc(func, order), order=order, sensitivity=sensitivity, s_span=s_span, solver=solver,
                                       atol=atol, rtol=rtol)
        self.nfe = self.defunc.nfe
        self.intloss = intloss
        self.u, self.controlled = None, False # datasets-control
        if sensitivity=='adjoint': self.adjoint = Adjoint(self.defunc, intloss)

        self._solver_checks(solver, sensitivity)

    def _solver_checks(self, solver, sensitivity):

        self.solver =  {'method': solver}

        if solver[:5] == "scipy" and solver not in SCIPY_SOLVERS:
            available_scipy_solvers = ", ".join(SCIPY_SOLVERS.keys())
            raise KeyError("Invalid Scipy Solver specified." +
                           " Supported Scipy Solvers are: " + available_scipy_solvers)

        elif solver in SCIPY_SOLVERS:
            warnings.warn(UserWarning("CUDA is not available with SciPy solvers."))

            if sensitivity == 'autograd':
                raise ValueError("SciPy Solvers do not work with autograd." +
                                 " Use adjoint sensitivity with SciPy Solvers.")

            self.solver = SCIPY_SOLVERS[solver]

    def _prep_odeint(self, x:torch.Tensor):
        self.s_span = self.s_span.to(x.device)

        # loss dimension detection routine; for CNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (not self.intloss is None) and self.sensitivity == 'autograd':
            excess_dims += 1

        # handle aux. operations required for some jacobian trace CNF estimators e.g Hutchinson's
        # as well as datasets-control set to DataControl module
        for name, module in self.defunc.named_modules():
            if hasattr(module, 'trace_estimator'):
                if module.noise_dist is not None: module.noise = module.noise_dist.sample((x.shape[0],))
                excess_dims += 1
            # datasets-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
            if hasattr(module, 'u'):
                self.controlled = True
                module.u = x[:, excess_dims:].detach()

        return x

    def forward(self, x:torch.Tensor):
        x = self._prep_odeint(x)
        switcher = {
            'autograd': self._autograd_forward,
            'adjoint': self._adjoint_forward,
            'torchdiffeq_adjoint': self._torchdiffeq_adjoint_forward
        }
        odeint = switcher.get(self.sensitivity)
        out = odeint(x)
        return out

    def trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        """Returns a data-flow trajectory at `s_span` points
        :param x: input data
        :type x: torch.Tensor
        :param s_span: collections of points to evaluate the function at e.g torch.linspace(0, 1, 100) for a 100 point trajectory
                       between 0 and 1
        :type s_span: torch.Tensor
        """
        x = self._prep_odeint(x)
        sol = torchdiffeq.odeint(self.defunc, x, s_span,
                                 rtol=self.rtol, atol=self.atol, **self.solver)
        return sol

    def backward_trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        raise NotImplementedError

    def _autograd_forward(self, x):
        self.defunc.intloss, self.defunc.sensitivity = self.intloss, self.sensitivity
        return torchdiffeq.odeint(self.defunc, x, self.s_span,
                                  rtol=self.rtol, atol=self.atol, **self.solver)[-1]

    def _adjoint_forward(self, x):
        return self.adjoint(self.defunc, x, self.s_span, rtol=self.rtol, atol=self.atol, **self.solver)
    
    def _torchdiffeq_adjoint_forward(self, x):
        return torchdiffeq.odeint_adjoint(self.defunc, x, self.s_span,
                                      rtol=self.rtol, atol=self.atol, **self.solver,
                                      adjoint_options=dict(norm=make_norm(x)))[-1]

    def trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        """Returns a datasets-flow trajectory at `s_span` points

        :param x: input datasets
        :type x: torch.Tensor
        :param s_span: collections of points to evaluate the function at e.g torch.linspace(0, 1, 100) for a 100 point trajectory
                       between 0 and 1
        :type s_span: torch.Tensor
        """
        x = self._prep_odeint(x)
        sol = torchdiffeq.odeint(self.defunc, x, s_span,
                                 rtol=self.rtol, atol=self.atol, **self.solver)
        return sol

    def sensitivity_trajectory(self, x:torch.Tensor, grad_output:torch.Tensor, 
                               s_span:torch.Tensor):
        assert self.sensitivity == 'adjoint', 'Sensitivity trajectory only available for `adjoint`'
        x = torch.autograd.Variable(x, requires_grad=True)
        sol = self(x)       
        adj0 = self.adjoint._init_adjoint_state(sol, grad_output)
        self.adjoint.flat_params = flatten(self.defunc.parameters())
        self.adjoint.func = self.defunc; self.adjoint.f_params = tuple(self.defunc.parameters())
        adj_sol = torchdiffeq.odeint(self.adjoint.adjoint_dynamics, adj0, s_span, 
               rtol=self.rtol, atol=self.atol, method=self.solver)
        return adj_sol


class NeuralSDE(NeuralDETemplate):
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
        for name, module in self.defunc.named_modules():
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
