import torch
import torch.nn as nn
from torchdyn.nn import DataControl, Augmenter
import pytorch_lightning as pl
from torchdyn.datasets import ToyDataset
from torch.utils.data import TensorDataset, DataLoader
import pytest

torch.manual_seed(123456789)


@pytest.fixture()
def moons_trainloader():
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    train = TensorDataset(X_train, y_train)
    trainloader = DataLoader(train, batch_size=len(X), shuffle=False)
    return trainloader


@pytest.fixture()
def small_mlp():
    net = nn.Sequential(nn.Linear(2, 64),
                        nn.Tanh(),
                        nn.Linear(64, 2)
                )
    return net


@pytest.fixture()
def small_dc_mlp():
    net = nn.Sequential(DataControl(),
                        nn.Linear(2, 64),
                        nn.Tanh(),
                        nn.Linear(64, 2)
                )
    return net


class TestLearner(pl.LightningModule):
    def __init__(self, model:nn.Module, trainloader):
        super().__init__()
        self.trainloader = trainloader
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return self.trainloader

@pytest.fixture()
def testlearner():
    return TestLearner


class TestIntegralLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, s, x):
        return x.norm(dim=1, p=2)


@pytest.fixture()
def testintloss():
    return TestIntegralLoss


@pytest.fixture
def moons_dataloader():
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    return X_train, data.DataLoader(train, batch_size=len(X), shuffle=False)

def rosenbrock(x):
    "a=1, b=100"
    x, y = x[...,:1], x[...,1:]
    return (1-x)**2 + 100*(y-x**2)**2

def quad(x):
    return (x-1)**2

def cubic(x):
    return (x)**3

def ackley(x):
    x = (x**2).sum(-1, keepdims=True)
    return -200*torch.exp(-0.02*torch.sqrt(x))

def bartels_conn(x):
    x, y = x[...,:1], x[...,1:]
    return (x**2+y**2+x*y).abs() + torch.sin(x).abs() + torch.cos(y).abs()
