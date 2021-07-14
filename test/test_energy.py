import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from torchdyn.models.energy import GNF, HNN, LNN
from torchdyn.datasets import ToyDataset


def test_stable_neural_de(testlearner):
    """Stable: basic functionality"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = GNF(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)))
    model = NeuralODE(f)
    t_span = torch.linspace(0, 1, 30)
    learn = testlearner(t_span, model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=5, max_epochs=5)
    trainer.fit(learn)

def test_hnn(testlearner):
    """HNN: basic functionality"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=32, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = HNN(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)))
    model = NeuralODE(f)
    t_span = torch.linspace(0, 1, 30)
    learn = testlearner(t_span, model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=5, max_epochs=5)
    trainer.fit(learn)

def test_lnn(testlearner):
    """LNN: basic functionality"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=32, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = LNN(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)))
    model = NeuralODE(f, solver='rk4')
    t_span = torch.linspace(0, 1, 30)
    learn = testlearner(t_span, model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
