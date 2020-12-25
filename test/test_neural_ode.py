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
from torchdyn.datasets import *
from torchdyn.models import *
from .utils import TestLearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_pretty_print():
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2))
    model = NeuralDE(f).to(device)
    print(model)

def test_work_without_settings():
    """Functionality: defining Neural DEs via default settings"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)
    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2))
    model = NeuralDE(f).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)

def test_neural_de_traj():
    """Vanilla NeuralDE sanity test"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2))
    model = NeuralDE(f,  solver='dopri5').to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)
    model.trajectory(X_train, s_span).detach().cpu()

def test_data_control():
    """Data-controlled NeuralDE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    f = nn.Sequential(DataControl(),
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 2))
    model = NeuralDE(f, solver='dopri5').to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)
    model.trajectory(X_train, s_span).detach().cpu()

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
