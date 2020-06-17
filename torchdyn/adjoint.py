"""
Adjoint template and variations of the adjoint technique
"""

import pdb
import torch
import torch.nn as nn
from torchdiffeq._impl.misc import _flatten
from torchdiffeq import odeint


class Adjoint(nn.Module):
    """Adjoint class template.

    :param integral: `True` if an *integral cost* (see **link alla pagina degli adj**) is specified
    :type integral: bool
    :param terminal: `True` if a *terminal cost* (see **link alla pagina degli adj**) is specified
    :type integral: bool
    :param return_traj: `True` if we want to return the whole adjoint trajectory and not only the final point (loss gradient)
    :type return_traj: bool
    """
    def __init__(self, integral:bool=False, terminal:bool=True, return_traj:bool=False):
        super().__init__()
        self.integral, self.terminal, self.return_traj = integral, terminal, return_traj
        self.autograd_func = self._define_autograd_adjoint()

    def adjoint_dynamics(self, s, adjoint_state):
        """ Define the vector field of the augmented adjoint dynamics(as in [link]) to be then integrated **backward**. An `Adjoint` object is istantiated into the `NeuralDE` if the adjoint method for back-propagation was selected.

        :param s: current depth
        :type s: float
        :param adjoint_state: tuple of four tensors constituting the *augmented adjoint state* to be integrated: `h` (hidden state of the neural ODE), `λ` (Lagrange multiplier), `μ` (loss gradient state), `s_adj` (adjoint state of the integration depth)
        :type adjoint_state: tuple of tensors
        """
        h, λ, μ, s_adj = adjoint_state[0:4]
        with torch.enable_grad():
            s = s.to(h.device).requires_grad_(True)
            h = h.requires_grad_(True) #.detach()
            f = self.func(s, h)
            dλds = torch.autograd.grad(f, h, -λ, allow_unused=True, retain_graph=True)[0]
            # dμds is a tuple! of all self.f_params groups
            dμds = torch.autograd.grad(f, self.f_params, -λ, allow_unused=True, retain_graph=True)
            if self.integral:
                g = self.cost(s, h, f)
                dgdh = torch.autograd.grad(g.mean(), h, allow_unused=True, retain_graph=True)[0]
                dλds = dλds - dgdh   
        ds_adjds = torch.tensor(0.).to(self.s_span)
        dμds = torch.cat([el.flatten() if el is not None else torch.zeros_like(p) for el, p in zip(dμds, self.f_params)]).to(dλds)
        return (f, dλds, dμds, ds_adjds)
  
    def _init_adjoint_state(self, sol, *grad_output):
        # check grad_output
        if self.terminal:
            λ0 = grad_output[-1][0]
        else:
            λ0 = torch.zeros_like(grad_output[-1][0])

        s_adj0 = torch.tensor(0.).to(self.s_span)
        μ0 = torch.zeros_like(self.flat_params)
        
        return (sol[-1], λ0, μ0, s_adj0)

    def _define_autograd_adjoint(self):
        class autograd_adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h0, flat_params, s_span):        
                with torch.no_grad():
                    sol = odeint(self.func, h0, self.s_span, rtol=self.rtol, atol=self.atol, 
                                 method=self.method, options=self.options)
                ctx.save_for_backward(self.s_span, self.flat_params, sol)
                sol = sol if self.return_traj else sol[-1]; return sol

            @staticmethod  
            def backward(ctx, *grad_output):
                s, flat_params, sol = ctx.saved_tensors
                self.f_params = tuple(self.func.parameters())
                with torch.no_grad():
                    adj0 = self._init_adjoint_state(sol, grad_output) 
                    adj_sol = odeint(self.adjoint_dynamics, adj0, self.s_span.flip(0), 
                                   rtol=self.rtol, atol=self.atol, method=self.method, options=self.options)
                λ = adj_sol[1]
                μ = adj_sol[2]
                return (λ, μ, None) 
        return autograd_adjoint
  
    def forward(self, func, h0, s_span, cost=None, rtol=1e-6, atol=1e-12, method='dopri5', options={}):
        if not isinstance(func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')
        self.flat_params = _flatten(func.parameters()) ; self.s_span = s_span
        self.func, self.cost = func, cost ; self.method, self.options = method, options
        self.atol, self.rtol = atol, rtol ; 
        sol = self.autograd_func.apply(h0, self.flat_params, self.s_span)
        return sol
