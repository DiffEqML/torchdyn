import pytest

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdyn.models import NeuralODE
from torchdyn.nn import DataControl, DepthCat
from torchdyn.nn.galerkin import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vector_fields = [nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2)),
                 nn.Sequential(DataControl(), nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
                 ]

@pytest.mark.parametrize('basis', [Fourier(3), VanillaRBF(3), GaussianRBF(3),
                                   MultiquadRBF(3), Polynomial(2), Chebychev(2)])
def test_default_run_gallinear(moons_trainloader, testlearner, basis):
    f = nn.Sequential(nn.Linear(2, 8),
                      nn.Tanh(),
                      DepthCat(1),
                      GalLinear(8, 2, expfunc=basis))
    model = NeuralODE(f)
    learn = testlearner(model, trainloader=moons_trainloader)
    trainer = pl.Trainer(min_epochs=100, max_epochs=100)
    trainer.fit(learn)

