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

"Experimental API for hybrid Neural DEs and continuous models applied to sequences -> [ODE-RNN, Neural CDE]"

import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl
import torchsde

from torchdyn.models import LSDEFunc


class HybridNeuralDE(nn.Module):
    def __init__(self, flow, jump, out, last_output=True, reverse=False):
        """ODE-RNN / LSTM / GRU"""
        super().__init__()
        self.flow, self.jump, self.out = flow, jump, out
        self.reverse, self.last_output = reverse, last_output

        # determine type of `jump` func
        # jump can be of two types:
        # either take hidden and element of sequence (e.g RNNCell)
        # or h, x_t and c (LSTMCell). Custom implementation assumes call
        # signature of type (x_t, h) and .hidden_size property
        if type(jump) == nn.modules.rnn.LSTMCell:
            self.jump_func = self._jump_latent_cell
        else:
            self.jump_func = self._jump_latent

    def forward(self, x):
        h = c = self._init_latent(x)
        Y = torch.zeros(x.shape[0], *h.shape).to(x)
        if self.reverse: x_t = x_t.flip(0)
        for t, x_t in enumerate(x):
            h, c = self.jump_func(x_t, h, c)
            h = self.flow(h)
            Y[t] = h
        Y = self.out(Y)
        return Y[-1] if self.last_output else Y

    def _init_latent(self, x):
        x = x[0]
        return torch.zeros(x.shape[0], self.jump.hidden_size).to(x.device)

    def _jump_latent(self, *args):
        x_t, h, c = args[:3]
        return self.jump(x_t, h), c

    def _jump_latent_cell(self, *args):
        x_t, h, c = args[:3]
        return self.jump(x_t, (h, c))


class LatentNeuralSDE(NeuralSDE, pl.LightningModule): # pragma: no cover
    def __init__(self, post_drift, diffusion, prior_drift, sigma, theta, mu, options,
                 noise_type, order, sensitivity, s_span, solver, atol, rtol, intloss):
        """Latent Neural SDEs."""

        super().__init__(drift_func=post_drift, diffusion_func=diffusion, noise_type=noise_type,
                         order=order, sensitivity=sensitivity, s_span=s_span, solver=solver,
                         atol=atol, rtol=rtol, intloss=intloss)

        self.defunc = LSDEFunc(f=post_drift, g=diffusion, h=prior_drift)
        self.defunc.noise_type, self.defunc.sde_type = noise_type, 'ito'
        self.options = options

        # p(y0).
        logvar = math.log(sigma ** 2. / (2. * theta))
        self.py0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.py0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=False)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

    def forward(self, eps: torch.Tensor, s_span=None):
        eps = eps.to(self.qy0_std)
        x0 = self.qy0_mean + eps * self.qy0_std

        qy0 = Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = kl_divergence(qy0, py0).sum(1).mean(0)  # KL(time=0).

        if s_span is not None:
            s_span_ext = s_span
        else:
            s_span_ext = self.s_span.cpu()

        zs, logqp = torchsde.sdeint(sde=self.defunc, x0=x0, s_span=s_span_ext,
                           rtol=self.rtol, atol=self.atol, logqp=True, options=self.options,
                           adaptive=self.adaptive, method=self.solver)

        logqp = logqp.sum(0).mean(0)
        log_ratio = logqp0 + logqp  # KL(time=0) + KL(path).

        return zs, log_ratio

    def sample_p(self, vis_span, n_sim, eps=None, bm=None, dt=0.01):
        eps = torch.randn(n_sim, 1).to(self.py0_mean).to(self.device) if eps is None else eps
        y0 = self.py0_mean + eps.to(self.device) * self.py0_std
        return torchsde.sdeint(self.defunc, y0, vis_span, bm=bm, method='srk', dt=dt, names={'drift': 'h'})

    def sample_q(self, vis_span, n_sim, eps=None, bm=None, dt=0.01):
        eps = torch.randn(n_sim, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps.to(self.device) * self.qy0_std
        return torchsde.sdeint(self.defunc, y0, vis_span, bm=bm, method='srk', dt=dt)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)
