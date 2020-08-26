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
    
class TestIntegralLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, s, x):
        return x.norm(dim=1, p=2)