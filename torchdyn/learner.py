"""
PyTorch Lightning Learner template to be used for training
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    """`Learner` following PyTorch Lightning templates. Handles training and validation routines. Refer to notebook
        tutorials as well as PyTorch Lightning documentation for best practices.
        
    :param model: model to train and validate
    :type model: nn.Module
    """
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x:torch.Tensor):
        """Returns self.model(x)
        :param x: input data
        :type x: torch.Tensor
        """
        return self.model(x)
    
    def training_step(self, batch:torch.Tensor, batch_idx:int):
        """Handles a training step with batch `batch`. User defined.
         :param x: input data
         :type x: torch.Tensor
         :param x: input data
         :type x: torch.Tensor
         """
        pass
    
    def configure_optimizers(self):
        """Configures the optimizers to utilize for training. User defined.
        """
        pass

    def train_dataloader(self):
        """Configures the train dataloader to utilize for training. User defined.
        """
        pass