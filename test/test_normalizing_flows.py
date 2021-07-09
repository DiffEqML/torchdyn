import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter
from torchdyn.models.cnf import CNF, hutch_trace, autograd_trace


def test_cnf_vanilla():
    device = torch.device('cpu')
    net = nn.Sequential(
            nn.Linear(2, 512),
            nn.ELU(),
            nn.Linear(512, 2)
        )
    defunc = CNF(net)
    nde = NeuralODE(defunc, solver='dopri5', atol=1e-5, rtol=1e-5, sensitivity='adjoint', return_t_eval=False)
    model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                          nde).to(device)
    x = torch.randn((512, 2)).to(device)
    out = model(x)[-1]
    assert out.shape[1] == x.shape[1] + 1

def test_hutch_vanilla():
    device = torch.device('cpu')
    net = nn.Sequential(
            nn.Linear(2, 512),
            nn.ELU(),
            nn.Linear(512, 2)
        )
    noise_dist = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    defunc = nn.Sequential(CNF(net, trace_estimator=hutch_trace, noise_dist=noise_dist))
    nde = NeuralODE(defunc, solver='dopri5', atol=1e-5, rtol=1e-5, sensitivity='adjoint', return_t_eval=False)
    model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                          nde).to(device)
    x = torch.randn((512, 2)).to(device)
    out = model(x)[-1]
    assert out.shape[1] == x.shape[1] + 1

def test_hutch_estimator_gauss_noise():
    noise_dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
    x_in = torch.randn((64, 2), requires_grad=True)
    m = nn.Sequential(nn.Linear(2, 32), nn.Softplus(), nn.Linear(32, 2))
    x_out = m(x_in)
    trJ = autograd_trace(x_out, x_in)
    hutch_trJ = torch.zeros(trJ.shape)
    for i in range(10000):
        x_out = m(x_in)
        eps = noise_dist.sample((64,))
        hutch_trJ += hutch_trace(x_out, x_in, noise=eps)
    assert (hutch_trJ / 10000 - trJ < 1e-1).all()
