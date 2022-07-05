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

import pytest
import torch
import torch.nn as nn
from torch.autograd import grad
import pytorch_lightning as pl
import torch.utils.data as data
from torchdyn.datasets import ToyDataset
from torchdyn.core import NeuralODE
from torchdyn.nn import GalLinear, GalConv2d, DepthCat, Augmenter, DataControl
from torchdyn.numerics import odeint, Euler

from functools import partial
import copy


if torch.cuda.is_available():
    devices = [torch.device("cuda:0"), torch.device("cpu")]
else:
    devices = [torch.device("cpu")]

vector_fields = [nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2)),
                 nn.Sequential(DataControl(), nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
                 ]
t_span = torch.linspace(0, 1, 30)


def test_repr(small_mlp):
    model = NeuralODE(small_mlp)
    assert type(model.__repr__()) == str and 'NFE' in model.__repr__()


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('vector_field', vector_fields)
def test_default_run(moons_trainloader, vector_field, testlearner, device):
    model = NeuralODE(vector_field, solver='dopri5', atol=1e-2, rtol=1e-2, sensitivity='interpolated_adjoint')
    learn = testlearner(t_span, model, trainloader=moons_trainloader)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(learn)


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def test_trajectory(moons_trainloader, small_mlp, testlearner, device):
    model = NeuralODE(small_mlp)
    learn = testlearner(t_span, model, trainloader=moons_trainloader)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(learn)

    x, _ = next(iter(moons_trainloader))
    trajectory = model.trajectory(x, t_span)
    assert len(trajectory) == 30


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def test_save(moons_trainloader, small_mlp, testlearner, device):
    model = NeuralODE(small_mlp, solver='euler')
    num_save = int(torch.randint(1, len(t_span)//2, [1]))  # random number of save points up to half as many as in tspan
    unique_inds = torch.unique(torch.randint(1, len(t_span), [num_save]))  # get that many indices and trim to unique
    save_at = t_span[unique_inds]
    save_at.sort()
    x, _ = next(iter(moons_trainloader))
    _, y_save = model(x, t_span, save_at)
    assert len(y_save) == len(save_at)

# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def test_dict_out_and_args(moons_trainloader, small_mlp, testlearner, device):

    def fun(t, x, args):
        inps = torch.cat([x["i1"], x["i2"]], dim=-1)
        outs = small_mlp(inps)
        return t, {"i1": outs[..., 0:1], "i2": outs[..., 1:2]}

    class DummyIntegrator(Euler):
        def __init__(self):
            super(DummyIntegrator, self).__init__()

        def step(self, f, x, t, dt, k1=None, args=None):
            _, x_sol = f(t, x, args)
            return None, x_sol, None

    x0 = {"i1": torch.rand(1, 1), "i2": torch.rand(1, 1)}
    model = NeuralODE(fun, solver=DummyIntegrator())
    _, y_save = model(x0, t_span)


@pytest.mark.skip(reason='Update to test saving and loading')
@pytest.mark.parametrize('device', devices)
def test_deepcopy(small_mlp, device):
    model = NeuralODE(small_mlp)
    x = torch.rand(1, 2)
    copy_before_forward = copy.deepcopy(model)
    assert type(copy_before_forward) == NeuralODE

    # do a forward+backward pass
    y = model(x)
    loss = y.sum()
    loss.backward()
    copy_after_forward = copy.deepcopy(model)
    assert type(copy_after_forward) == NeuralODE


@pytest.mark.skip(reason='clean up to new API')
def test_augmenter_func_is_trained():
    """Test if augment function is trained without explicit definition"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DataControl(),
                      nn.Linear(12, 64),
                      nn.Tanh(),
                      nn.Linear(64, 6))
    model = nn.Sequential(Augmenter(augment_idx=1, augment_func=nn.Linear(2, 4)),
                          NeuralODE(f, solver='dopri5')
                         ).to(device)
    learn = TestLearner(t_span, model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    p = torch.cat([p.flatten() for p in model[0].parameters()])
    trainer.fit(learn)
    p_after = torch.cat([p.flatten() for p in model[0].parameters()])
    assert (p != p_after).any()


# TODO
@pytest.mark.skip(reason='clean up to new API')
def test_augmented_data_control():
    """Data-controlled NeuralODE with IL-Augmentation"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)

    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DataControl(),
                     nn.Linear(12, 64),
                     nn.Tanh(),
                     nn.Linear(64, 6))

    model = nn.Sequential(Augmenter(augment_idx=1, augment_func=nn.Linear(2, 4)),
                          NeuralODE(f, solver='dopri5')
                         ).to(device)
    learn = TestLearner(t_span, model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    trainer.fit(learn)


# TODO
@pytest.mark.skip(reason='clean up to new API')
def test_vanilla_galerkin():
    """Vanilla Galerkin (MLP) Neural ODE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)

    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DepthCat(1),
                      GalLinear(6, 64, basisfunc=Fourier(5)),
                      nn.Tanh(),
                      DepthCat(1),
                      GalLinear(64, 6, basisfunc=Polynomial(2)))

    model = nn.Sequential(Augmenter(augment_idx=1, augment_func=nn.Linear(2, 4)),
                          NeuralODE(f, solver='dopri5')
                         ).to(device)
    learn = TestLearner(t_span, model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)


# TODO
@pytest.mark.skip(reason='clean up to new API')
def test_vanilla_conv_galerkin():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """Vanilla Galerkin (CNN 2D) Neural ODE"""
    X = torch.randn(12, 1, 28, 28).to(device)

    f = nn.Sequential(DepthCat(1),
                      GalConv2d(1, 12, kernel_size=3, padding=1, basisfunc=Fourier(3)),
                      nn.Tanh(),
                      DepthCat(1),
                      GalConv2d(12, 1, kernel_size=3, padding=1, basisfunc=Fourier(3)))

    model = nn.Sequential(NeuralODE(f, solver='dopri5')).to(device)
    model(X)


# TODO
@pytest.mark.skip(reason='clean up to new API')
def test_2nd_order():
    """2nd order (MLP) Galerkin Neural ODE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)

    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DepthCat(1),
                      nn.Linear(5, 64),
                      nn.Tanh(),
                      DepthCat(1),
                      nn.Linear(65, 2))

    model = nn.Sequential(Augmenter(augment_idx=1, augment_func=nn.Linear(2, 2)),
                          NeuralODE(f, solver='dopri5', order=2)
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)


# https://github.com/DiffEqML/torchdyn/issues/118
def test_arg_ode():
    """Test sensitivity through NeuralODE solutions of a functools.partial vector field"""
    l = nn.Linear(1, 1)

    class TFunc(nn.Module):
        def __init__(self, l):
            super().__init__()
            self.l = l
        def forward(self, t, x, u, v, z, args={}):
            return self.l(x + u + v + z)

    tfunc = TFunc(l)

    u = v = z = torch.randn(1, 1)
    f = partial(tfunc.forward, u=u, v=v, z=z)
    x0 = torch.randn(1, 1, requires_grad=True)
    t_eval, sol1 = odeint(f, x0, torch.linspace(0, 5, 10), solver='euler')

    odeprob = NeuralODE(f, 'euler', sensitivity='interpolated_adjoint', optimizable_params=tfunc.parameters())
    t_eval, sol2 = odeprob(x0, t_span=torch.linspace(0, 5, 10))

    assert (sol1==sol2).all()
    grad(sol2.sum(), x0)