import torch.utils.data as data
from utils import TestLearner

import torchdyn; from torchdyn.models import *; from torchdyn.datasets import *
import torch ; import torch.nn as nn
from torch.distributions import *

def test_work_without_settings():
    """Functionality: defining Neural DEs via default settings"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    trainer = pl.Trainer(min_epochs=10, max_epochs=30)
    trainer.fit(learn) 
    
def test_neural_de_traj():
    """Vanilla NeuralDE sanity test"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    trainer = pl.Trainer(min_epochs=10, max_epochs=30)
    trainer.fit(learn) 
    s_span = torch.linspace(0, 1, 100)
    trajectory = model.trajectory(X_train, s_span).detach().cpu()

def test_data_control():
    """Data-controlled NeuralDE"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    trainer = pl.Trainer(min_epochs=10, max_epochs=30)

    trainer.fit(learn) 
    s_span = torch.linspace(0, 1, 100)
    trajectory = model.trajectory(X_train, s_span).detach().cpu()

def test_augmenter_func_is_trained():
    """Test if augment function is trained without explicit definition"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    trainer = pl.Trainer(min_epochs=10, max_epochs=30)

    p = torch.cat([p.flatten() for p in model[0].parameters()])
    trainer.fit(learn) 
    p_after = torch.cat([p.flatten() for p in model[0].parameters()])
    assert (p != p_after).any()
    
def test_augmented_data_control():
    """Data-controlled NeuralDE with IL-Augmentation"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='spirals', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    trainer = pl.Trainer(min_epochs=10, max_epochs=30)

    trainer.fit(learn) 

    
if __name__ == '__main__':  
    print(f'Testing NeuralDE no settings...')
    test_work_without_settings()
    print(f'Testing NeuralDE .trajectory method...')
    test_neural_de_traj()
    print(f'Testing DataControl...')
    test_data_control()
    print(f'Testing whether Augmenter function receives weight updates...')
    test_augmenter_func_is_trained()
    print(f'Testing Augmenter + DataControl...')
    test_augmented_data_control()