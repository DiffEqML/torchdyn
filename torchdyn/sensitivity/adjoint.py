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

"""
Adjoint template and variations of the adjoint technique
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


def flatten(iterable):
    sol = torch.zeros(1) if len(iterable) == 0 else torch.cat([el.contiguous().flatten() for el in iterable])
    return sol

class ODEAdjointFunc:
    """ Define the vector field of the augmented adjoint dynamics to be then integrated **backward**. An `Adjoint` object is istantiated into the `NeuralDE` if the adjoint method for back-propagation was selected.
    :param s: current depth
    :type s: float
    :param adjoint_state: tuple of four tensors constituting the *augmented adjoint state* to be integrated: `h` (hidden state of the neural ODE), `λ` (Lagrange multiplier), `μ` (loss gradient state), `s_adj` (adjoint state of the integration depth)
    :type adjoint_state: tuple of tensors
    """
    def __init__(self, f, g=None):
        self.f, self.g = f, g

    def solve_forward(self, s, z):
        return self.f(s, z)

    def solve_backward(self, s, adjoint_state):
        z, λ, μ, s_adj = adjoint_state[0:4]
        # temporarily removed
        # s = s.to(h.device).requires_grad_(True)  
        with torch.set_grad_enabled(True):
            z = z.requires_grad_(True)
            dzds = self.f(s, z)
            dλds = torch.autograd.grad(dzds, z, -λ, allow_unused=True, retain_graph=True)[0]
            dμds = torch.autograd.grad(dzds, tuple(self.f_params), -λ, allow_unused=True, retain_graph=True)
            if not self.g is None:
                dgdh = torch.autograd.grad(self.g(s, z).sum(), z, allow_unused=True, retain_graph=True)[0]
                dλds = dλds - dgdh
            ds_adjds = torch.tensor(0.).to(z)
            # Safety checks for `None` gradients
            dμds = torch.cat([el.flatten() if el is not None else torch.zeros_like(p) for el, p in zip(dμds, self.f_params)]).to(z)
        return (dzds, dλds, dμds, ds_adjds)
    

class Adjoint(nn.Module):
    """Adjoint class template.
    :param intloss: `nn.Module` specifying the integral loss term
    :type intloss: nn.Module
    """
    def __init__(self, f, g=None):
        super().__init__()
        self._adjoint_func = ODEAdjointFunc(f, g)

    def _init_adjoint_state(self, sol, *grad_output):
        λ0 = grad_output[-1][0]
        s_adj0 = torch.tensor(0.).to(λ0)
        μ0 = torch.zeros_like(flatten(self._adjoint_func.f_params))
        return (sol[-1], λ0, μ0, s_adj0)

    def _wrap_func_autograd(self, s_span, rtol, atol, method, options):
        class autograd_adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h0, flat_params, s_span):
                sol = odeint(self._adjoint_func.solve_forward, h0, s_span, rtol=rtol, atol=atol,
                             method=method, options=options)
                ctx.save_for_backward(s_span, sol)
                return sol[-1]

            @staticmethod
            def backward(ctx, *grad_output):
                s, sol = ctx.saved_tensors
                adj0 = self._init_adjoint_state(sol, grad_output)
                adj_sol = odeint(self._adjoint_func.solve_backward, adj0, s_span.flip(0),
                               rtol=rtol, atol=atol, method=method, options=options)
                λ = adj_sol[1]
                μ = adj_sol[2]
                return (λ, μ, None)
        return autograd_adjoint

    
    def forward(self, func, h0, s_span, rtol=1e-4, atol=1e-4, method='dopri5', options={}):
        if not isinstance(func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')
        h0 = h0.requires_grad_(True)
        self._adjoint_func.f_params = list(func.parameters())

        flat_params = flatten(self._adjoint_func.f_params)
        self._wrapped_adjoint_func = self._wrap_func_autograd(s_span, rtol, atol, method, options)
        sol = self._wrapped_adjoint_func.apply(h0, flat_params, s_span)
        return sol


def find_f_params(module):
    assert isinstance(module, nn.Module)
    if getattr(module, '_is_replica', False):
        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples
        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())
