import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl
import torchsde

from torchdyn.models import LSDEFunc

class LatentNeuralSDE(NeuralSDE, pl.LightningModule): # pragma: no cover
    def __init__(self, post_drift, diffusion, prior_drift, sigma, theta, mu, options,
                 noise_type, order, sensitivity, s_span, solver, atol, rtol, intloss):

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
        """
        :param s_span: (optional) Series span -- can pass extended version for additional regularization
        :param eps: Noise sample
        """

        eps = eps.to(self.qy0_std)
        x0 = self.qy0_mean + eps * self.qy0_std

        qy0 = Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = kl_divergence(qy0, py0).sum(1).mean(0)  # KL(time=0).

        # Expand s_span to penalize out-of-datasets region and spread uncertainty -- moved
        # s_span_ext = torch.cat((torch.tensor([0.0]), self.s_span.to('cpu'), torch.tensor([2.0])))

        if s_span is not None:
            s_span_ext = s_span
        else:
            s_span_ext = self.s_span.cpu()

        zs, logqp = sdeint(sde=self.defunc, x0=x0, s_span=s_span_ext,
                           rtol=self.rtol, atol=self.atol, logqp=True, options=self.options,
                           adaptive=self.adaptive, method=self.solver)

        logqp = logqp.sum(0).mean(0)
        log_ratio = logqp0 + logqp  # KL(time=0) + KL(path).

        return zs, log_ratio

    def sample_p(self, vis_span, n_sim, eps=None, bm=None, dt=0.01):
        """
        :param vis_span:
        :param n_sim:
        :param eps:
        :param bm:
        :param dt:
        """
        eps = torch.randn(n_sim, 1).to(self.py0_mean).to(self.device) if eps is None else eps
        y0 = self.py0_mean + eps.to(self.device) * self.py0_std
        return torchsde.sdeint(self.defunc, y0, vis_span, bm=bm, method='srk', dt=dt, names={'drift': 'h'})

    def sample_q(self, vis_span, n_sim, eps=None, bm=None, dt=0.01):
        """
        :param vis_span:
        :param n_sim:
        :param eps:
        :param bm:
        :param dt:
        """
        eps = torch.randn(n_sim, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps.to(self.device) * self.qy0_std
        return torchsde.sdeint(self.defunc, y0, vis_span, bm=bm, method='srk', dt=dt)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)
