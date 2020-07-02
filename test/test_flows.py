import sys
sys.path.append('..')
import torchdyn; from torchdyn.models import *; from torchdyn.datasets import *
import torch ; import torch.nn as nn
from torch.distributions import *

def test_cnf_vanilla():
    device = torch.device('cpu')
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
    device = torch.device('cpu')
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
    assert (hutch_trJ / 10000 - trJ < 1e-2).all()

if __name__ == '__main__':
    print(f'Testing regular CNF with autograd trace...')
    test_cnf_vanilla()
    print(f'Testing regular CNF with Hutch. estimator...')
    test_hutch_vanilla()
    print(f'Checking accuracy of Hutch. estimator (gauss epsilon) vs autograd true trace...')
    test_hutch_estimator_gauss_noise() 


    
    
