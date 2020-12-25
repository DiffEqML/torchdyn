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

from torchdyn.models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1415112413244349)


def test_strato_sde(testlearner):
    """Test vanilla Stratonovich Neural SDE"""
    X_train, trainloader = moons_dataloader
  
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
    learn = testlearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)


def test_ito_sde(testlearner):
    """Test vanilla Ito Neural SDE"""
    X_train, trainloader = moons_dataloader
    
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
    learn = testlearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 0.1, 100)
    model.trajectory(X_train, s_span).detach().cpu()


def test_data_control(testlearner):
    """Data-controlled Neural SDE"""
    X_train, trainloader = moons_dataloader

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
    learn = testlearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 0.1, 100)
    model.trajectory(X_train, s_span).detach().cpu()


def test_depth_cat(testlearner):
    """DepthCat Neural SDE"""

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
    learn = testlearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 0.1, 100)
    model.trajectory(X_train, s_span).detach().cpu()
