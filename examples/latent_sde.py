"""A partial Re-implementation of Xuechen Li's work (https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py)"""


import os
import math
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Laplace

from torchdyn.models import LatentNeuralSDE, LinearScheduler, EMAMetric
from torchsde import BrownianPath


class IrregularSineDataset(Dataset):
    def __init__(self, batch_size, num_batches):
        ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_ = self.make_irregular_sine_data()
        self.array = ys.view(-1).unsqueeze(0).repeat(batch_size*num_batches, 1)
        self.ts = ts
        self.ts_ext = ts_ext
        self.ts_vis = ts_vis
        self.ys = ys

    def __len__(self): return len(self.array)
    def __getitem__(self, i): return self.array[i]
    def s_span(self): return self.ts
    def s_ext_span(self): return self.ts_ext
    def v_span(self): return self.ts_vis
    def x_sample(self): return self.ys

    @staticmethod
    def make_irregular_sine_data():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])
        with torch.no_grad():
            ts_ = np.sort(np.random.uniform(low=0.4, high=1.6, size=16))
            ts_ext_ = np.array([0.] + list(ts_) + [2.0])
            ts_vis_ = np.linspace(0., 2.0, 300)
            ys_ = np.sin(ts_ * (2. * math.pi))[:, None] * 0.8

            ts = torch.tensor(ts_).float().to(device)
            ts_ext = torch.tensor(ts_ext_).float()
            ts_vis = torch.tensor(ts_vis_).float()
            ys = torch.tensor(ys_).float().to(device)

            return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)


class FFunc(pl.LightningModule):
    """Posterior drift."""
    def __init__(self):
        super(FFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )

    def forward(self, t, y):
        if t.dim() == 0:
            t = float(t) * torch.ones_like(y)
        # Positional encoding in transformers; must use `t`, since the posterior is likely inhomogeneous.
        inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        return self.net(inp)


class HFunc(pl.LightningModule):
    """Prior drift"""
    def __init__(self, theta=1.0, mu=0.0):
        super(HFunc, self).__init__()
        self.theta = nn.Parameter(torch.tensor([[theta]]), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)

    def forward(self, t, y):
        return self.theta * (self.mu - y)


class GFunc(pl.LightningModule):
    """Diffusion"""
    def __init__(self, sigma=0.5):
        super(GFunc, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([[sigma]]), requires_grad=False)

    def forward(self, t, y):
        return self.sigma.repeat(y.size(0), 1)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        sigma, theta, mu = 0.5, 1.0, 0.0
        options = {'trapezoidal_approx': False}

        self.f_func = FFunc()
        self.h_func = HFunc(theta=theta, mu=mu)
        self.g_func = GFunc(sigma=sigma)

        self.s_span = IrregularSineDataset(1, 1).s_span()

        self.lsde = LatentNeuralSDE(post_drift=self.f_func, diffusion=self.g_func, prior_drift=self.h_func,
                                    sigma=sigma, theta=theta, mu=mu,
                                    noise_type='diagonal', order=1, sensitivity='autograd', s_span=self.s_span,
                                    solver='srk', atol=1e-3, rtol=1e-3, intloss=None, options=options)

    def forward(self, eps: torch.Tensor, s_span=None):
        """
        :param: Noise sample
        """
        zs, log_ratio = self.lsde(eps, s_span)
        zs = zs.squeeze()
        #

        return zs, log_ratio


class Learner(pl.LightningModule):
    def __init__(self, train_path):
        super().__init__()
        self.model = Model()

        dataset = IrregularSineDataset(1, 1)
        self.vis_span = dataset.v_span()
        self.x_sample = dataset.x_sample()
        self.s_span = dataset.s_span()
        self.s_ext_span = dataset.s_ext_span()

        self.train_path = train_path
        self.logp_metric = EMAMetric()
        self.log_ratio_metric = EMAMetric()
        self.loss_metric = EMAMetric()
        self.kl_scheduler = LinearScheduler(iters=10)

        self.scale = 0.05

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x, y = torch.split(batch, split_size_or_sections=1, dim=0)
        x = batch
        eps = torch.randn(batch.shape[0], 1)

        zs, log_ratio = self.model(eps=eps, s_span=self.s_ext_span)
        zs = zs[1:-1]

        likelihood = Laplace(loc=zs, scale=self.scale)

        # Bad Hack just in this case where every tensor in batch is identical
        logp = likelihood.log_prob(x.mean(dim=0).unsqueeze(1).to(self.device)).sum(dim=0).mean(dim=0)
        loss = -logp + log_ratio * self.kl_scheduler()

        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()
        self.logp_metric.step(logp)
        self.log_ratio_metric.step(log_ratio)
        self.loss_metric.step(loss)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def on_epoch_end(self, vis_n_sim=1024):

        img_path = os.path.join(train_dir, f'global_step_{self.current_epoch}.png')
        ylims = (-1.75, 1.75)
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
        percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
        fill_color = '#9ebcda'
        mean_color = '#4d004b'
        num_samples = len(sample_colors)
        vis_idx = np.random.permutation(vis_n_sim)

        eps = torch.randn(vis_n_sim, 1)
        bm = BrownianPath(t0=self.vis_span[0], w0=torch.zeros(vis_n_sim, 1))

        # -- Not used -- From show_prior option in original implementation
        # zs = self.model.sample_p(vis_span=self.vis_span, n_sim=vis_n_sim, eps=eps, bm=bm).squeeze()
        # ts_vis_, zs_ = self.vis_span.cpu().numpy(), zs.cpu().numpy()
        # zs_ = np.sort(zs_, axis=1)

        zs = self.model.lsde.sample_q(vis_span=self.vis_span, n_sim=vis_n_sim, eps=eps, bm=bm).squeeze()
        samples = zs[:, vis_idx]
        s_span_vis_ = self.vis_span.cpu().detach().numpy()
        zs_ = zs.cpu().detach().numpy()
        samples_ = samples.cpu().detach().numpy()

        zs_ = np.sort(zs_, axis=1)

        with torch.no_grad():

            plt.subplot(frameon=False)

            for alpha, percentile in zip(alphas, percentiles):
                idx = int((1 - percentile) / 2. * vis_n_sim)
                zs_bot_, zs_top_ = zs_[:, idx], zs_[:, -idx]
                plt.fill_between(s_span_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

            plt.plot(s_span_vis_, zs_.mean(axis=1), color=mean_color)

            for j in range(num_samples):
                plt.plot(s_span_vis_, samples_[:, j], color=sample_colors[j], linewidth=1.0)

            num, ds = 12, 0.12
            s, x = torch.meshgrid(
                [torch.linspace(0.2, 1.8, num), torch.linspace(-1.5, 1.5, num)]
            )

            s, x = s.reshape(-1, 1).to(self.device), x.reshape(-1, 1).to(self.device)

            ftx = self.model.lsde.defunc.f(s=s, x=x)
            ftx = ftx.cpu().reshape(num, num)

            ds = torch.zeros(num, num).fill_(ds)
            dx = ftx * ds
            ds_, dx_, = ds.cpu().detach().numpy(), dx.cpu().detach().numpy()
            s_, x_ = s.cpu().detach().numpy(), x.cpu().detach().numpy()

            plt.quiver(s_, x_, ds_, dx_, alpha=0.3, edgecolors='k', width=0.0035, scale=50)

            # Data.
            plt.scatter(self.s_span.cpu().numpy(), self.x_sample.cpu().numpy(), marker='x', zorder=3, color='k', s=35)

            plt.ylim(ylims)
            plt.xlabel('$t$')
            plt.ylabel('$Y_t$')
            plt.tight_layout()
            plt.savefig(img_path, dpi=400)
            plt.close()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)

        return [optimizer], [scheduler]

    def train_dataloader(self):

        batch_size = 512
        n_batches = 50

        return DataLoader(IrregularSineDataset(batch_size=batch_size, num_batches=n_batches), batch_size=batch_size)


train_dir = os.path.join('images', 'ts_ext')
trainer = pl.Trainer(gpus=0, max_epochs=200)
trainer.fit(Learner(train_dir))








