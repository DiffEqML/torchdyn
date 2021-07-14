import pytest

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from torchdyn.nn import *

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
    t_span = torch.linspace(0, 1, 30)
    model = NeuralODE(f, solver='rk4')
    learn = testlearner(t_span, model, trainloader=moons_trainloader)
    trainer = pl.Trainer(min_epochs=5, max_epochs=10)
    trainer.fit(learn)

