import torch
import torch.nn as nn
import torchdiffeq
from torchdyn.core.neuralde import NeuralODE

def hook_backward_gradients(module, grad_input, grad_output): # pragma: no cover
    module.grad_output = grad_output

class HypernetDepth(nn.Module): # pragma: no cover
    def __init__(self, g, baseline):
        super().__init__()
        self.g = g
        self.baseline = baseline

    def forward(self, x):
        return torch.cat([torch.zeros(1).to(x), torch.abs(self.g(x) + 1)[0]])
    
    def grad(self, x):
        fxS, dLdxS = self.fxS.mean(), self.grad_output[0].mean()
        for p in self.g.parameters():
            dSdp = torch.autograd.grad(torch.abs(self.g(x) + 1).mean(), p)[0]
            with torch.no_grad():
                if hasattr(p, 'grad'): p.grad += fxS * dLdxS * dSdp
                else: p.grad = fxS * dLdxS * dSdp

    
class AdaptiveDepthNeuralODE(NeuralODE): # pragma: no cover
    """Adaptive-Depth Neural ODE

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
                       intloss=None,
                       depthfunc=None):
        super().__init__(func, order, sensitivity, s_span, solver, atol, rtol, intloss)

        self.depthfunc = HypernetDepth(depthfunc, 2)
        # pass self to depthfunc for hooks         
        self.depthfunc.register_backward_hook(hook_backward_gradients)
        
    def _prep_odeint(self, x:torch.Tensor):
        # determines s_span dynamically. Does not support different
        # s_spans for different datasets samples (yet)
        self.s_span = self.depthfunc(x)
        self.S = self.s_span[-1]
        return super()._prep_odeint(x)
    
    def forward(self, x:torch.Tensor):       
        # hooks f(x(S)) for gradients
        # TO DO: use hooks
        out = super().forward(x); 
        self.depthfunc.fxS = self.defunc( self.S, out)
        return out
    
    def trajectory(self, x:torch.Tensor, s_span:torch.Tensor):
        """Returns a datasets-flow trajectory at `s_span` points

        :param x: input datasets
        :type x: torch.Tensor
        :param s_span: collections of points to evaluate the function at e.g torch.linspace(0, 1, 100) for a 100 point trajectory
                       between 0 and 1
        :type s_span: torch.Tensor
        """
        x = super()._prep_odeint(x)
        sol = torchdiffeq.odeint(self.defunc, x, s_span,
                                 rtol=self.rtol, atol=self.atol, method=self.solver)
        return sol
                
