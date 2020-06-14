import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from test_utils import TestLearner

import sys
sys.path.append('..')
from torchdyn.models import *; from torchdyn.data_utils import *
from torchdyn import *

def test_work_without_settings():
    """Functionality: defining Neural DEs via `default` settings"""
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)    
    f = DEFunc(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(), 
            nn.Linear(64, 2)))
    model = NeuralDE(f, settings={}).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_nb_epochs=10, max_nb_epochs=30, verbose=False, show_progress_bar=False)
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
    settings = {'type':'classic', 'controlled':False, 'solver':'dopri5'}
    f = DEFunc(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(), 
            nn.Linear(64, 2)))
    model = NeuralDE(f, settings).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_nb_epochs=10, max_nb_epochs=30, verbose=False, show_progress_bar=False)
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
    settings = {'type':'classic', 'controlled':True, 'solver':'adaptive_heun'}
    f = DEFunc(nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(), 
            nn.Linear(64, 2)))
    model = NeuralDE(f, settings).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_nb_epochs=10, max_nb_epochs=30, verbose=False, show_progress_bar=False)
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
    settings = {'type':'classic', 'controlled':True, 'solver':'rk4', 's_span':torch.linspace(0, 1, 100)}
    f = DEFunc(nn.Sequential(nn.Linear(12, 64),
                             nn.Tanh(), 
                             nn.Linear(64, 6)))
    model = nn.Sequential(Augmenter(augment_idx=1, augment_func=nn.Linear(2, 4)),
                          NeuralDE(f, settings)
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_nb_epochs=10, max_nb_epochs=30, verbose=False, show_progress_bar=False)
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
    settings = {'type':'classic', 'controlled':True, 'solver':'rk4', 's_span':torch.linspace(0, 1, 100)}
    f = DEFunc(nn.Sequential(nn.Linear(12, 64),
                             nn.Tanh(), 
                             nn.Linear(64, 6)))
    model = nn.Sequential(Augmenter(augment_idx=1, augment_func=nn.Linear(2, 4)),
                          NeuralDE(f, settings)
                         ).to(device)
    learn = TestLearner(model, trainloader=trainloader)
    trainer = pl.Trainer(min_nb_epochs=10, max_nb_epochs=30, verbose=False, show_progress_bar=False)
    trainer.fit(learn) 

    
if __name__ == '__main__':  
    test_work_without_settings()
    test_neural_de_traj()
    test_data_control()
    test_augmenter_func_is_trained()
    test_augmented_data_control()