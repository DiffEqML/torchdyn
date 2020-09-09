import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl
import torchsde
from torchsde import sdeint_adjoint


class SDEFunc(nn.Module):
    def __init__(self, f, g, order=1):
        super().__init__()  
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func = f, g
        self.fnfe, self.gnfe = 0, 0

    def forward(self, s, x):
        pass
    
    def f(self, s, x):
        """Posterior drift."""
        self.fnfe += 1
        return self.f_func(x)
    
    def g(self, s, x):
        """Diffusion"""
        self.gnfe += 1
        return self.g_func(x).diag_embed()


class LSDEFunc(pl.LightningModule):
    def __init__(self, f, g, h, order=1):
        super().__init__()
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func, self.h_func = f, g, h
        self.fnfe, self.gnfe, self.hnfe = 0, 0, 0

    def forward(self, s, x):
        pass

    def h(self, s, x):
        """ Prior drift
        :param s:
        :param x:
        """
        self.hnfe += 1
        return self.h_func(t=s, y=x)

    def f(self, s, x):
        """Posterior drift.
        :param s:
        :param x:
        """
        self.fnfe += 1
        return self.f_func(t=s, y=x)

    def g(self, s, x):
        """Diffusion.
        :param s:
        :param x:
        """
        self.gnfe += 1
        return self.g_func(t=s, y=x)
