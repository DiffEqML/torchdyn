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
import pytorch_lightning as pl
import torch.utils.data as data
from torchdyn.datasets import ToyDataset
from torchdyn.models import NeuralODE
from torchdyn.nn.galerkin import GalLinear, GalConv2d
from torchdyn import DepthCat, Augmenter, DataControl

import copy


if torch.cuda.is_available():
    devices = [torch.device("cuda:0"), torch.device("cpu")]
else:
    devices = [torch.device("cpu")]

vector_fields = [nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2)),
                 nn.Sequential(DataControl(), nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
                 ]


def test_repr(small_mlp):
    model = NeuralODE(small_mlp)
    assert type(model.__repr__()) == str and 'NFE' in model.__repr__()


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('vector_field', vector_fields)
def test_default_run(moons_trainloader, vector_field, testlearner, device):
    model = NeuralODE(vector_field)
    learn = testlearner(model, trainloader=moons_trainloader)
    trainer = pl.Trainer(min_epochs=500, max_epochs=500)
    trainer.fit(learn)
    assert trainer.logged_metrics['train_loss'] < 1e-1


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def test_trajectory(moons_trainloader, small_mlp, testlearner, device):
    model = NeuralODE(small_mlp)
    learn = testlearner(model, trainloader=moons_trainloader)
    trainer = pl.Trainer(min_epochs=500, max_epochs=500)
    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)

    x, _ = next(iter(moons_trainloader))
    trajectory = model.trajectory(x, s_span)
    assert len(trajectory) == 100


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


# TODO
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
                          NeuralDE(f, solver='dopri5')
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    p = torch.cat([p.flatten() for p in model[0].parameters()])
    trainer.fit(learn)
    p_after = torch.cat([p.flatten() for p in model[0].parameters()])
    assert (p != p_after).any()


# TODO
@pytest.mark.skip(reason='clean up to new API')
def test_augmented_data_control():
    """Data-controlled NeuralDE with IL-Augmentation"""
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
                          NeuralDE(f, solver='dopri5')
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
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
                          NeuralDE(f, solver='dopri5')
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
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

    model = nn.Sequential(NeuralDE(f, solver='dopri5')).to(device)
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
                          NeuralDE(f, solver='dopri5', order=2)
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
