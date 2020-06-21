import sys
sys.path.append('..')
import torchdyn; from torchdyn.models import *; from torchdyn.datasets import *
import torch ; import torch.nn as nn
from torch.distributions import *


def test_vanilla():
    device = torch.device('cuda')
    net = nn.Sequential(
            nn.Linear(2, 512),
            nn.ELU(),
            nn.Linear(512, 2)
        )
    defunc = CNF(net)
    nde = NeuralDE(defunc, solver='dopri5', s_span=torch.linspace(0, 1, 2), atol=1e-5, rtol=1e-5, sensitivity='adjoint')
    model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                          nde).to(device)
    x = torch.randn((512, 2)).to(device)
    out = model(x)
    assert out.shape[1] == x.shape[1] + 1
    
def test_hutch_vanilla():
    device = torch.device('cuda')
    net = nn.Sequential(
            nn.Linear(2, 512),
            nn.ELU(),
            nn.Linear(512, 2)
        )
    noise_dist = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    defunc = nn.Sequential(CNF(net, trace_estimator=hutch_trace, noise_dist=noise_dist))
    
    nde = NeuralDE(defunc, solver='dopri5', s_span=torch.linspace(0, 1, 2), atol=1e-5, rtol=1e-5, sensitivity='adjoint')
    model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                          nde).to(device)  
    x = torch.randn((512, 2)).to(device)
    out = model(x)
    assert out.shape[1] == x.shape[1] + 1
    
