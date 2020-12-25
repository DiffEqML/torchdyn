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

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import *
from torchdyn.datasets import *
from torchdyn.models import *
from .utils import TestLearner


def test_strato_sde():
    """Test vanilla Stratonovich Neural SDE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
    g = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))

    model = NeuralSDE(f, g,
                    noise_type='diagonal',
                    sde_type='stratonovich',
                    sensitivity='adjoint',
                    s_span=torch.linspace(0, 0.1, 100),
                    solver='euler_heun',
                    atol=1e-4,
                    rtol=1e-4).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)

def test_ito_sde():
    """Test vanilla Ito Neural SDE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
    g = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))

    model = NeuralSDE(f, g,
                    noise_type='diagonal',
                    sde_type='ito',
                    sensitivity='adjoint',
                    s_span=torch.linspace(0, 0.1, 100),
                    solver='euler',
                    atol=0.0001,
                    rtol=0.0001).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 0.1, 100)
    model.trajectory(X_train, s_span).detach().cpu()

def test_data_control():
    """Data-controlled Neural SDE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DataControl(), nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
    g = nn.Sequential(DataControl(), nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))

    model = NeuralSDE(f, g,
                    noise_type='diagonal',
                    sde_type='ito',
                    sensitivity='adjoint',
                    s_span=torch.linspace(0, 0.1, 100),
                    solver='euler',
                    atol=0.0001,
                    rtol=0.0001).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    trainer.fit(learn)
    s_span = torch.linspace(0, 0.1, 100)
    model.trajectory(X_train, s_span).detach().cpu()

def test_depth_cat():
    """DepthCat Neural SDE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DepthCat(1), nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 2))
    g = nn.Sequential(DepthCat(1), nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 2))

    model = NeuralSDE(f, g,
                    noise_type='diagonal',
                    sde_type='ito',
                    sensitivity='adjoint',
                    s_span=torch.linspace(0, 0.1, 100),
                    solver='euler',
                    atol=0.0001,
                    rtol=0.0001).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 0.1, 100)
    model.trajectory(X_train, s_span).detach().cpu()