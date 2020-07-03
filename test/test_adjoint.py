import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from test_utils import TestLearner

import sys
sys.path.append('..')
from torchdyn.models import *; from torchdyn.data_utils import *
from torchdyn import *

def test_adjoint_autograd():
    """Compare ODE Adjoint vs Autograd gradients, s := [0, 1], adaptive-step"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)    
    settings = {'type':'classic', 'controlled':False, 'solver':'dopri5', 
                'backprop_style':'adjoint', 'rtol':1e-5, 'atol':1e-5}
    f = DEFunc(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(), 
            nn.Linear(64, 2)))
    model = NeuralDE(f, settings).to(device)
    x, y = next(iter(trainloader)) 
    # adjoint gradients
    y_hat = model(x)   
    loss = nn.CrossEntropyLoss()(y_hat, y)
    loss.backward()
    adj_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    # autograd gradients
    model.zero_grad()
    model.st['backprop_style']= 'autograd'
    y_hat = model(x)   
    loss = nn.CrossEntropyLoss()(y_hat, y)
    loss.backward()
    bp_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert (torch.abs(bp_grad - adj_grad) <= 1e-4).all()