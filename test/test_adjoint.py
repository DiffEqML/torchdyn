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

from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
from torchdyn.models import NeuralODE
from torchdyn.datasets import ToyDataset
from torchdyn import Augmenter


batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1415112413244349)

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

    model = NeuralODE(f, solver='dopri5', sensitivity='adjoint', atol=1e-5, rtol=1e-8).to(device)
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
    assert (torch.abs(bp_grad - adj_grad) <= 1e-3).all(), f'Gradient error: {torch.abs(bp_grad - adj_grad).sum()}'


def test_integral_adjoint_integral_autograd(testintloss):
    """Compare ODE Adjoint vs Autograd gradients (with integral loss), s := [0, 1], adaptive-step"""
    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64,2))
    aug = Augmenter(1, 1)
    torch.manual_seed(0)
    model_autograd = NeuralODE(f, solver='dopri5', sensitivity='autograd', atol=1e-5, rtol=1e-5, intloss=testintloss()).to(device)
    torch.manual_seed(0)
    model_adjoint = NeuralODE(f, solver='dopri5', sensitivity='adjoint', atol=1e-5, rtol=1e-5, intloss=testintloss()).to(device)


    torch.manual_seed(0)
    x = torch.randn(batch_size, 2).to(device)
    x = x.requires_grad_(True)
    a = model_autograd(aug(x))
    loss = a[:, 0].sum()
    loss.backward()
    g_autograd = deepcopy(x.grad)

    torch.manual_seed(0)
    x = torch.randn(batch_size, 2).to(device)
    x = x.requires_grad_(True)
    a = model_adjoint(x)
    loss = 0.*a.sum()
    loss.backward()
    g_adjoint= deepcopy(x.grad)
    assert torch.abs(g_autograd - g_adjoint).norm(dim=1, p=2).mean() < 5e-3


def test_distributed_loss():
    assert True