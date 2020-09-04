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
    return torch.cat([el.contiguous().flatten() for el in iterable])

class Adjoint(nn.Module):
    """Adjoint class template.

    :param intloss: `nn.Module` specifying the integral loss term
    :type intloss: nn.Module
    """
    def __init__(self, intloss = None):
        super().__init__()

        self.intloss = intloss ; self.autograd_func = self._define_autograd_adjoint()

    def adjoint_dynamics(self, s, adjoint_state):
        """ Define the vector field of the augmented adjoint dynamics to be then integrated **backward**. An `Adjoint` object is istantiated into the `NeuralDE` if the adjoint method for back-propagation was selected.

        :param s: current depth
        :type s: float
        :param adjoint_state: tuple of four tensors constituting the *augmented adjoint state* to be integrated: `h` (hidden state of the neural ODE), `λ` (Lagrange multiplier), `μ` (loss gradient state), `s_adj` (adjoint state of the integration depth)
        :type adjoint_state: tuple of tensors
        """
        h, λ, μ, s_adj = adjoint_state[0:4]
        with torch.set_grad_enabled(True):
            s = s.to(h.device).requires_grad_(True)
            h = h.requires_grad_(True)
            f = self.func(s, h)
            dλds = torch.autograd.grad(f, h, -λ, allow_unused=True, retain_graph=True)[0]
            # dμds is a tuple! of all self.f_params groups
            dμds = torch.autograd.grad(f, self.f_params, -λ, allow_unused=True, retain_graph=True)
            if not self.intloss is None:
                g = self.intloss(s, h)
                dgdh = torch.autograd.grad(g.sum(), h, allow_unused=True, retain_graph=True)[0]
                dλds = dλds - dgdh
        ds_adjds = torch.tensor(0.).to(self.s_span)

        # `None` safety check necessary for cert. applications e.g. Stable with bias on out layer
        dμds = torch.cat([el.flatten() if el is not None else torch.zeros_like(p) for el, p in zip(dμds, self.f_params)]).to(dλds)

        return (f, dλds, dμds, ds_adjds)

    def _init_adjoint_state(self, sol, *grad_output):
        λ0 = grad_output[-1][0]
        s_adj0 = torch.tensor(0.).to(self.s_span)
        μ0 = torch.zeros_like(self.flat_params)
        return (sol[-1], λ0, μ0, s_adj0)

    def _define_autograd_adjoint(self):
        class autograd_adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h0, flat_params, s_span):
                sol = odeint(self.func, h0, self.s_span, rtol=self.rtol, atol=self.atol,
                             method=self.method, options=self.options)
                ctx.save_for_backward(self.s_span, self.flat_params, sol)
                return sol[-1]

            @staticmethod
            def backward(ctx, *grad_output):
                s, flat_params, sol = ctx.saved_tensors
                self.f_params = tuple(self.func.parameters())
                adj0 = self._init_adjoint_state(sol, grad_output)
                adj_sol = odeint(self.adjoint_dynamics, adj0, self.s_span.flip(0),
                               rtol=self.rtol, atol=self.atol, method=self.method, options=self.options)
                λ = adj_sol[1]
                μ = adj_sol[2]
                return (λ, μ, None)
        return autograd_adjoint

    def forward(self, func, h0, s_span, rtol=1e-4, atol=1e-4, method='dopri5', options={}):
        if not isinstance(func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')
        self.flat_params = flatten(func.parameters()) ; self.s_span = s_span
        self.func = func; self.method, self.options = method, options
        self.atol, self.rtol = atol, rtol ;
        h0 = h0.requires_grad_(True)
        sol = self.autograd_func.apply(h0, self.flat_params, self.s_span)
        return sol
