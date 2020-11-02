import pytorch_lightning as pl
import torch
import torch.nn as nn

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
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return self.trainloader

class TestIntegralLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, s, x):
        return x.norm(dim=1, p=2)
