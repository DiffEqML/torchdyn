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

# Test adjoint and perform a rough benchmarking of wall-clock time

import time
from copy import deepcopy
import logging

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data
import torchdiffeq
from torchdyn.core import ODEProblem, NeuralODE
from torchdyn.nn import Augmenter
from torchdyn.datasets import ToyDataset
from torchdyn.numerics import VanDerPol


batch_size = 128
torch.manual_seed(1415112413244349)

t_span = torch.linspace(0, 1, 100)

logger = logging.getLogger("out")


# TODO(numerics): log wall-clock times and other torch.grad tests
# TODO(bug): `tsit5` + `adjoint` peak error
@pytest.mark.parametrize('sensitivity', ['adjoint', 'interpolated_adjoint'])
@pytest.mark.parametrize('solver', ['dopri5', 'tsit5'])
@pytest.mark.parametrize('stiffness', [0.1, 0.5])
@pytest.mark.parametrize('interpolator', [None])
def test_odeint_adjoint(sensitivity, solver, interpolator, stiffness):

    f = VanDerPol(stiffness)
    x = torch.randn(1024, 2, requires_grad=True)
    
    prob = ODEProblem(f, sensitivity=sensitivity, interpolator=interpolator, solver=solver, atol=1e-4, rtol=1e-4, atol_adjoint=1e-4, rtol_adjoint=1e-4)
    t0 = time.time()
    t_eval, sol_torchdyn = prob.odeint(x, t_span)
    t_end1 = time.time() - t0

    t0 = time.time()
    sol_torchdiffeq = torchdiffeq.odeint_adjoint(f, x, t_span, method='dopri5', atol=1e-4, rtol=1e-4)
    t_end2 = time.time() - t0

    logger.info(f"Fwd times: {t_end1:.3f}, {t_end2:.3f}")

    true_sol = torchdiffeq.odeint_adjoint(f, x, t_span, method='dopri5', atol=1e-9, rtol=1e-9)

    t0 = time.time()
    grad1 = torch.autograd.grad(sol_torchdyn[-1].sum(), x)[0]
    t_end1 = time.time() - t0

    t0 = time.time()
    grad2 = torch.autograd.grad(sol_torchdiffeq[-1].sum(), x)[0]
    t_end2 = time.time() - t0

    logger.info(f"Bwd times: {t_end1:.3f}, {t_end2:.3f}")

    grad_true = torch.autograd.grad(true_sol[-1].sum(), x)[0]

    err1 = (grad1-grad_true).abs().sum(1)
    err2 = (grad2-grad_true).abs().sum(1)
    assert (err1 <= 1e-3).all() and (err1.mean() <= err2.mean())


@pytest.mark.parametrize('stiffness', [0.5])
@pytest.mark.parametrize('sensitivity', ['adjoint', 'interpolated_adjoint'])
def test_odeint_adjoint_intloss(stiffness, sensitivity):
    f = VanDerPol(stiffness)
    def reg_term(t, x):
        return 0.1*x[:,:1]

    x = torch.randn(1024, 2, requires_grad=True)

    # solve with autograd augmentation, fixed-step solver for discrete-adj consistency
    node = NeuralODE(f, sensitivity='autograd', solver='rk4', integral_loss=reg_term)
    x0 = torch.cat([torch.zeros(x.shape[0], 1), x], 1)

    t_eval, sol_torchdyn = node(x0, t_span)
    loss = sol_torchdyn[-1,:,0].sum()
    loss.backward()
    grad_autograd = x.grad
    x.grad = 0*x.grad

    # solve with integral loss without aug
    node = NeuralODE(f, sensitivity=sensitivity, solver='dopri5', interpolator='4th', 
                    atol=1e-4, rtol=1e-4, atol_adjoint=1e-5, rtol_adjoint=1e-5, integral_loss=reg_term)

    t_eval, sol_torchdyn = node(x, t_span)
    (0.*sol_torchdyn[-1].sum()).backward()
    grad_adj = x.grad

    assert (grad_autograd-grad_adj).abs().mean() <= 1e-4


@pytest.mark.skip("")
def test_odeint_adjoint_trained():
    """Compare ODE Adjoint vs Autograd gradients, s := [0, 1], adaptive-step"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2))

    model = NeuralODE(f, solver='dopri5', sensitivity='adjoint', atol=1e-5, rtol=1e-8)
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

