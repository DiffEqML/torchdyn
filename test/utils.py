import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import sys
import torchdyn; from torchdyn.models import *; from torchdyn.datasets import *
import torch ; import torch.nn as nn
from torch.distributions import *

class TestLearner(pl.LightningModule):
    def __init__(self, model:nn.Module, trainloader):
        super().__init__()
        self.trainloader = trainloader
        self.model = model

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
        return self.trainloader