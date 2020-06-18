import torch
import torch.nn as nn
import torchdiffeq
import pytorch_lightning as pl
from ..adjoint import Adjoint
from .._internals import compat_check

defaults = {'type':'classic', 'controlled':False, 'augment':False, # model
            'backprop_style':'autograd', 'cost':None, # training 
            's_span':torch.linspace(0, 1, 2), 'solver':'rk4', 'atol':1e-3, 'rtol':1e-4, # solver params
            'return_traj':False} 

class NeuralDE(pl.LightningModule):
    """General Neural DE template

    :param func: function parametrizing the vector field.
    :type func: nn.Module
    :param settings: specifies parameters of the Neural DE. 
    :type settings: dict
    """
    def __init__(self, func:nn.Module, settings:dict):
        super().__init__()
        defaults.update(settings)
        compat_check(defaults)
        self.st = defaults ; self.defunc, self.defunc.func_type = func, self.st['type']
        self.defunc.controlled = self.st['controlled']
        self.s_span, self.return_traj = self.st['s_span'], self.st['return_traj']
        
        # check if integral
        flag = (self.st['backprop_style'] == 'integral_adjoint')
        self.adjoint = Adjoint(flag)
        
    def forward(self, x:torch.Tensor):
        return self._odesolve(x)    

    def _odesolve(self, x:torch.Tensor):
        # TO DO: implement adaptive_depth check, insert here
        
        # assign control input and augment if necessary 
        if self.defunc.controlled: self.defunc.u = x 
        self.s_span = self.s_span.to(x)
        
        switcher = {
        'autograd': self._autograd,
        'integral_autograd': self._integral_autograd,
        'adjoint': self._adjoint,
        'integral_adjoint': self._integral_adjoint
        }
        odeint = switcher.get(self.st['backprop_style'])
        sol = odeint(x) if self.st['return_traj'] else odeint(x)[-1]
        return sol

    def trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        """Returns a data-flow trajectory at `s_span` points

        :param x: input data
        :type x: torch.Tensor
        :param s_span: collections of points to evaluate the function at e.g torch.linspace(0, 1, 100) for a 100 point trajectory
                       between 0 and 1
        :type s_span: torch.Tensor
        """
        if self.defunc.controlled: self.defunc.u = x             
        sol = torchdiffeq.odeint(self.defunc, x, s_span,
                                 rtol=self.st['rtol'], atol=self.st['atol'], method=self.st['solver'])        
        return sol
    
    def backward_trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        assert self.adjoint, 'Propagating backward dynamics only possible with Adjoint systems'
        # register hook
        if self.defunc.controlled: self.defunc.u = x      
        # set new s_span
        self.adjoint.s_span = s_span ; x = x.requires_grad_(True)
        sol = self(x)
        sol.sum().backward()
        return sol.grad

    def _autograd(self, x):
        return torchdiffeq.odeint(self.defunc, x, self.s_span, rtol=self.st['rtol'], 
                                  atol=self.st['atol'], method=self.st['solver']) 
    def _adjoint(self, x):
        return torchdiffeq.odeint_adjoint(self.defunc, x, self.s_span, rtol=self.st['rtol'], 
                                          atol=self.st['atol'], method=self.st['solver'])
    def _integral_adjoint(self, x):
        assert self.st['cost'], 'Cost nn.Module needs to be specified for integral adjoint'
        return self.adjoint(self.defunc, x, self.s_span, cost=self.st['cost'],
                            rtol=self.st['rtol'], atol=self.st['atol'], method=self.st['solver'])
    
    def _integral_autograd(self, x):
        assert self.st['cost'], 'Cost nn.Module needs to be specified for integral adjoint'
        ξ0 = 0.*torch.ones(1).to(x.device)
        ξ0 = ξ0.repeat(x.shape[0]).unsqueeze(1)
        x = torch.cat([x,ξ0], 1)
        return torchdiffeq.odeint(self._integral_autograd_defunc, x, self.s_span,
                                rtol=self.st['rtol'], atol=self.st['atol'],method=self.st['solver'])

    def _integral_autograd_defunc(self, s, x):
        x = x[:, :-1]
        dxds = self.defunc(s, x)
        dξds = self.settings['cost'](s, x, dxds).repeat(x.shape[0]).unsqueeze(1)
        return torch.cat([dxds,dξds],1)
            
    def __repr__(self):
        npar = sum([p.numel() for p in self.defunc.parameters()])
        return f"Neural DE\tType: {self.st['type']}\tControlled: {self.st['controlled']}\
        \nSolver: {self.st['solver']}\tIntegration interval: {self.st['s_span'][0]} to {self.st['s_span'][-1]}\
        \nCost: {self.st['cost']}\tReturning trajectory: {self.st['return_traj']}\
        \nTolerances: relative {self.st['rtol']} absolute {self.st['atol']}\
        \nFunction parametrizing vec. field:\n {self.defunc}\
        \n# parameters {npar}"
