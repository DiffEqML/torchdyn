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
    Contains various utilities for `odeint` and numerical methods. Various norms, step size initialization, event callbacks for hybrid systems, vmapped matrix-Jacobian products and some
    additional goodies.
"""
import attr
import torch
import torch.nn as nn
from torch.distributions import Exponential
from torchcde import CubicSpline, hermite_cubic_coefficients_with_backward_differences


def make_norm(state):
    state_size = state.numel()
    def norm_(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(hairer_norm(y), hairer_norm(adj_y))
    return norm_


def hairer_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def init_step(f, f0, x0, t0, order, atol, rtol):
    scale = atol + torch.abs(x0) * rtol
    d0, d1 = hairer_norm(x0 / scale), hairer_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=x0.dtype, device=x0.device)
    else:
        h0 = 0.01 * d0 / d1

    x_new = x0 + h0 * f0
    f_new = f(t0 + h0, x_new)
    d2 = hairer_norm((f_new - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=x0.dtype, device=x0.device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))
    dt = torch.min(100 * h0, h1).to(t0)
    return dt


@torch.no_grad()
def adapt_step(dt, error_ratio, safety, min_factor, max_factor, order):
    if error_ratio == 0: return dt * max_factor
    if error_ratio < 1: min_factor = torch.ones_like(dt)
    exponent = torch.tensor(order, dtype=dt.dtype, device=dt.device).reciprocal()
    factor = torch.min(max_factor, torch.max(safety / error_ratio ** exponent, min_factor))
    return dt * factor


def dense_output(sol, t_sol, t_eval, return_spline=False):
    t_sol = t_sol.to(sol)
    spline_coeff = hermite_cubic_coefficients_with_backward_differences(t_sol, sol.permute(1, 0, 2))
    sol_spline = CubicSpline(t_sol, spline_coeff)
    sol_eval = torch.stack([sol_spline.evaluate(t) for t in t_eval])
    if return_spline:
        return sol_eval, sol_spline
    return sol_eval


class EventState:
    def __init__(self, evid):
        self.evid = evid

    def __ne__(self, other):
        return sum([a_ != b_ for a_, b_ in zip(self.evid, other.evid)])


@attr.s
class EventCallback(nn.Module):
    "Basic callback for hybrid differential equations. Must define an event condition and a state-jump"
    def __attrs_post_init__(self):
        super().__init__()

    def check_event(self, t, x):
        raise NotImplementedError

    def jump_map(self, t, x):
        raise NotImplementedError


@attr.s
class StochasticEventCallback(nn.Module):
    def __attrs_post_init__(self):
        super().__init__()
        self.expdist = Exponential(1)

    def initialize(self, x0):
        self.s = self.expdist.sample(x0.shape[:1])

    def check_event(self, t, x):
        raise NotImplementedError

    def jump_map(self, t, x):
        raise NotImplementedError
        
class RootLogger(object):
    def __init__(self):
        self.data = {'geval': [], 'z': [], 'dz': [], 'iteration': [], 'alpha': [], 'phi': []}

    def log(self, logged_data):
        self.data.update(**logged_data)

    def permanent_log(self, logged_data):
        for key in self.data.keys():
            self.data.update({key: list(self.data[key] + logged_data[key])})


class WrapFunc(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, t, x): return self.f(x)

