import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from utils import TestLearner, TestIntegralLoss

from torchdyn.models import *; from torchdyn.datasets import *
from torchdyn import *
from copy import deepcopy

def test_adjoint_autograd():
    """Compare ODE Adjoint vs Autograd gradients, s := [0, 1], adaptive-step"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)    
    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(), 
            nn.Linear(64, 2))
    
    model = NeuralDE(f, solver='dopri5', sensitivity='adjoint', atol=1e-5, rtol=1e-5).to(device)
    x, y = next(iter(trainloader)) 
    # adjoint gradients
    y_hat = model(x)   
    loss = nn.CrossEntropyLoss()(y_hat, y)
    loss.backward()
    adj_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    # autograd gradients
    model.zero_grad()
    model.sensitivity = 'autograd'
    y_hat = model(x)   
    loss = nn.CrossEntropyLoss()(y_hat, y)
    loss.backward()
    bp_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert (torch.abs(bp_grad - adj_grad) <= 1e-4).all(), f'Gradient error: {torch.abs(bp_grad - adj_grad).sum()}'
    
    

def test_integral_adjoint_integral_autograd():
    """Compare ODE Adjoint vs Autograd gradients (with integral loss), s := [0, 1], adaptive-step"""
    bs = 1000
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64,2))

    torch.manual_seed(0)
    aug = Augmenter(1, 1)
    torch.manual_seed(0)
    model_autograd = NeuralDE(f, solver='dopri5', sensitivity='autograd', atol=1e-5, rtol=1e-5, intloss=TestIntegralLoss()).to(device)
    torch.manual_seed(0)
    model_adjoint = NeuralDE(f, solver='dopri5', sensitivity='adjoint', atol=1e-5, rtol=1e-5, intloss=TestIntegralLoss()).to(device)
        
    torch.manual_seed(0)
    x = torch.randn(bs, 2).to(device)
    x = x.requires_grad_(True)
    a = model_autograd(aug(x))
    loss = a[:, 0].sum()
    loss = loss.backward()
    g_autograd = deepcopy(x.grad)
    
    torch.manual_seed(0)
    x = torch.randn(bs, 2).to(device)
    x = x.requires_grad_(True)
    a = model_adjoint(x)
    loss = 0.*a.sum()
    loss = loss.backward()
    g_adjoint= deepcopy(x.grad)
    
    assert torch.abs(g_autograd - g_adjoint).norm(dim=1, p=2).mean() < 1e-4