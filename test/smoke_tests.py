import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import sys


def train_traj_neural_de():
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    class Learner(pl.LightningModule):
        def __init__(self, model:nn.Module, settings:dict={}):
            super().__init__()
            defaults.update(settings)
            self.settings = defaults
            self.model = model
            self.c = 0

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch      
            y_hat = self.model(x)   
            loss = nn.CrossEntropyLoss()(y_hat, y)
            logs = {'train_loss': loss}
            return {'loss': loss, 'log': logs}   

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.005)

        def train_dataloader(self):
            return trainloader
        
    settings = {'type':'classic', 'controlled':False, 'solver':'dopri5'}

    f = DEFunc(nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(), 
            nn.Linear(64, 2)))

    model = NeuralDE(f, settings).to(device)
    
    learn = Learner(model)
    trainer = pl.Trainer(min_nb_epochs=10, max_nb_epochs=30)
    trainer.fit(learn)
    
    s_span = torch.linspace(0,1,100)
    trajectory = model.trajectory(X_train, s_span).detach().cpu()
    print("Test done")
    
if __name__ == '__main__':
  
    sys.path.append('..')
    from torchdyn.models import *; from torchdyn.data_utils import *
    from torchdyn import *
    train_traj_neural_de()