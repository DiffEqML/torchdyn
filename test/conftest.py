import pytest

import torch.utils.data as data
from torchdyn.datasets import *
from torchdyn.models import *
from .utils import TestIntegralLoss, TestLearner


@pytest.fixture
def moons_dataloader():
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    return X_train, data.DataLoader(train, batch_size=len(X), shuffle=False)
