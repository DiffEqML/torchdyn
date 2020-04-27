import torch
import torch.nn as nn
import torchdiffeq
import pytorch_lightning as pl
from ..adjoint import Adjoint

defaults = {'type':'classic', 'controlled':False, 'augment':False, # model
            'backprop_style':'autograd', 'cost':None, # training 
            's_span':torch.linspace(0, 1, 2), 'solver':'rk4', 'atol':1e-3, 'rtol':1e-4, # solver params
            'return_traj':False} 

class NeuralDE(pl.LightningModule):
    """General Neural DE template

    :param integral: `True` if an *integral cost* (see **link alla pagina degli adj**) is specified
    :type integral: bool
    :param return_traj: `True` if we want to return the whole adjoint trajectory and not only the final point (loss gradient)
    :type return_traj: bool
    """
    def __init__(self, func:nn.Module, settings:dict):
        super().__init__()
        #compat_check(settings)
        defaults.update(settings)
        self.settings = defaults ; self.defunc, self.defunc.func_type = func, settings['type']
        self.defunc.controlled = self.settings['controlled']
        self.s_span, self.return_traj = self.settings['s_span'], self.settings['return_traj']
        
        # check if integral
        flag = (self.settings['backprop_style'] == 'integral_adjoint')
        self.adjoint = Adjoint(flag)
        
    def forward(self, x:torch.Tensor):
        return self._odesolve(x)    

    def _odesolve(self, x:torch.Tensor):
        st = self.settings
        # TO DO: implement adaptive_depth check, insert here
        
        # assign control input and augment if necessary 
        if self.defunc.controlled: self.defunc.u = x 
        if st['augment']: self._augment(x)
        self.s_span = self.s_span.to(x)
        
        switcher = {
        'autograd': self._autograd,
        'integral_AD': self._integral_autograd,
        'adjoint': self._adjoint,
        'integral_adjoint': self._integral_adjoint
        }
        odeint = switcher.get(st['backprop_style'])
        sol = odeint(x) if st['return_traj'] else odeint(x)[-1]
        return sol

    def trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        """The adjoint is basically the adjoint

        :param integral: `True` if an *integral cost* (see **link alla pagina degli adj**) is specified
        :type integral: bool
        :param return_traj: `True` if we want to return the whole adjoint trajectory and not only the final point (loss gradient)
        :type return_traj: bool
        """
        st = self.settings 
        if self.defunc.controlled: self.defunc.u = x      
        # zero augmentation 
        if st['augment']: self._augment(x)           
        sol = torchdiffeq.odeint(self.defunc, x, s_span,
                                 rtol=st['rtol'], atol=st['atol'], method=st['solver'])        
        return sol
    
    # TO DO
    def backward_trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        assert self.adjoint, 'Propagating backward dynamics only possible with Adjoint systems'
        st = self.settings 
        # register hook
        if self.defunc.controlled: self.defunc.u = x      
        # augmentation 
        if st['augment']: self._augment(x) 
        # set new s_span
        self.adjoint.s_span = s_span ; x = x.requires_grad_(True)
        sol = self(x)
        sol.sum().backward()
        return sol.grad

    def _autograd(self, x):
        st = self.settings
        return torchdiffeq.odeint(self.defunc, x, self.s_span,
                                    rtol=st['rtol'], atol=st['atol'], method=st['solver']) 
    def _adjoint(self, x):
        st = self.settings
        return torchdiffeq.odeint_adjoint(self.defunc, x, self.s_span,
                                rtol=st['rtol'], atol=st['atol'], method=st['solver'])
    def _integral_adjoint(self, x):
        st = self.settings
        assert st['cost'], 'Cost nn.Module needs to be specified for integral adjoint'
        return self.adjoint(self.defunc, x, self.s_span, cost=st['cost'],
                            rtol=st['rtol'], atol=st['atol'], method=st['solver'])
    
    def _integral_autograd(self, x):
        st = self.settings
        assert st['cost'], 'Cost nn.Module needs to be specified for integral adjoint'
        ξ0 = 0.*torch.ones(1).to(x.device)
        ξ0 = ξ0.repeat(x.shape[0]).unsqueeze(1)
        x = torch.cat([x,ξ0],1)
        return torchdiffeq.odeint(self._integral_autograd_defunc, x, self.s_span,
                                rtol=st['rtol'], atol=st['atol'],method=st['solver'])

    def _integral_autograd_defunc(self, s, x):
        x = x[:,:-1]
        dxds = self.defunc(s, x)
        dξds = self.settings['cost'](s, x).repeat(x.shape[0]).unsqueeze(1)
        return torch.cat([dxds,dξds],1)
            
    def __repr__(self):
        return f"Neural DE\n{self.settings}"