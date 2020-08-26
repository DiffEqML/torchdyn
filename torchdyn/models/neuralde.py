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

import torch
import torch.nn as nn
import torchdiffeq
import pytorch_lightning as pl
from .defunc import DEFunc
from torchdyn.sensitivity.adjoint import Adjoint
from .._internals import compat_check

class NeuralDE(pl.LightningModule):
    """General Neural DE class

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
        super().__init__()
        #compat_check(defaults)
        # TO DO: remove controlled from input args
        self.defunc, self.order = DEFunc(func, order), order
        self.sensitivity, self.s_span, self.solver = sensitivity, s_span, solver
        self.nfe = self.defunc.nfe
        self.rtol, self.atol = rtol, atol
        self.intloss = intloss
        self.u, self.controlled = None, False # data-control
        
        if sensitivity=='adjoint': self.adjoint = Adjoint(self.intloss);
           
    def _prep_odeint(self, x:torch.Tensor):
        self.s_span = self.s_span.to(x)
             
        # loss dimension detection routine; for CNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (not self.intloss is None) and self.sensitivity == 'autograd':
            excess_dims += 1
                
        # handle aux. operations required for some jacobian trace CNF estimators e.g Hutchinson's
        # as well as data-control set to DataControl module
        for name, module in self.defunc.named_modules():
            if hasattr(module, 'trace_estimator'):
                if module.noise_dist is not None: module.noise = module.noise_dist.sample((x.shape[0],))  
                excess_dims += 1
                
        # data-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC 
        # TO DO: merge the named_modules loop for perf
        for name, module in self.defunc.named_modules():
            if hasattr(module, 'u'): 
                self.controlled = True
                module.u = x[:, excess_dims:].detach()
                   
        return x  

    def forward(self, x:torch.Tensor):  
        x = self._prep_odeint(x)        
        switcher = {
            'autograd': self._autograd,
            'adjoint': self._adjoint,
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
                                 rtol=self.rtol, atol=self.atol, method=self.solver)
        return sol
    
    def backward_trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        raise NotImplementedError

    def reset(self):
        self.nfe, self.defunc.nfe = 0, 0

    def _autograd(self, x):
        self.defunc.intloss, self.defunc.sensitivity = self.intloss, self.sensitivity

        if self.intloss == None:
            return torchdiffeq.odeint(self.defunc, x, self.s_span, rtol=self.rtol,
                                  atol=self.atol, method=self.solver)[-1]
        else:
            return torchdiffeq.odeint(self.defunc, x, self.s_span,
                                      rtol=self.rtol, atol=self.atol, method=self.solver)[-1]

    def _adjoint(self, x):
        return self.adjoint(self.defunc, x, self.s_span, rtol=self.rtol, atol=self.atol, method=self.solver)
    
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
    