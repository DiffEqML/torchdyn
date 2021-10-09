from os import X_OK
import torch
import torch.nn as nn
from torchvision.datasets import MNIST

import sys
sys.path.append('../../../')
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchdyn.core import NeuralODE
from torchdyn.models import CNF, vmapped_revmode_trace
from torchdyn.nn import Augmenter

from functorch import vmap


train_dataset = MNIST('../../../../data/MNIST', train=True, download=True)
test_dataset = MNIST('../../../../data/MNIST', train=False, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)

CHANNELS = [1 ,  1,  4,  4,  2,  2,  8,  8,  4,  4]
WIDTHS = [28, 28, 14, 14, 14, 14,  7,  7,  7,  7]


def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


class DEFuncBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(n_channels, 64, 3),
                            nn.Conv2d(64, 64, 3),
                            nn.Conv2d(64, 64, 3),
                            nn.Conv2d(64, n_channels, 3)
                            )

        self.act = nn.Softplus()

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(0) # when vmapping, we lose the first dim
        for conv in self.layers[:-1]:
            x = self.act(conv(x))
        return self.layers[-1](x)


class MNISTCNF(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Create NeuralODE blocks
        ndes = []
        for i in range(8):
            defunc = CNF(DEFuncBlock(CHANNELS[i]), trace_estimator=vmapped_revmode_trace)
            nde = NeuralODE(defunc, 
                            solver='rk4', 
                            sensitivity='autograd')
            
            # def. and append NeuralODE block
            ndes = ndes + [nde]

        self.ndes = nn.Sequential(*ndes)
        #self.bn1 = nn.BatchNorm2d(1)
        #self.bn2 = nn.BatchNorm2d(4)
        
        #for p in self.ndes[-1].parameters(): torch.nn.init.zeros_(p)
            
        #self.fl = nn.Flatten()
        
    def forward(self, x):
        #x = self.bn1(x)
        t_span = torch.linspace(0, 1, 4)
        x = Augmenter(augment_dims=1, order='first')(x)
        
        x = self.ndes[0](x, t_span)
        x = self.ndes[1](x, t_span)
        x = squeeze(x, 2)                   # first squeeze
        #x = self.bn2(x)
        x = self.ndes[2](x, t_span)
        x = self.ndes[3](x, t_span)
        out_0, x = x[:,:2,:,:], x[:,2:,:,:]  # first slice
        x = self.ndes[4](x, t_span)
        x = self.ndes[5](x, t_span)
        x = squeeze(x, 2)                    # second squeeze
        x = self.ndes[6](x, t_span)
        x = self.ndes[7](x, t_span)
        out_1, x = x[:,:4,:,:], x[:, 4:,:,:] # second slice
        x = self.ndes[8](x, t_span)
        out_2 = self.ndes[9](x, t_span)
        
        out = torch.cat([
            self.fl(out_0),
            self.fl(out_1),
            self.fl(out_2)], 1)
        return out
    
    def sample(self, x):
        for nde in self.ndes: nde.s_span = torch.linspace(1, 0, 4)
        in_0, in_1, in_2 = x[:,:392], x[:,392:392+196], x[:,-196:]
        in_0, in_1, in_2 =  nn.Unflatten(1, (2, 14, 14))(in_0), nn.Unflatten(1, (4, 7, 7))(in_1), nn.Unflatten(1, (4, 7, 7))(in_2),
        
        x = self.ndes[9](in_2)
        x = self.ndes[8](x)
        x = torch.cat([in_1, x], 1)
        x = self.ndes[7](x)
        x = self.ndes[6](x)   
        x = unsqueeze(x, 2)
        x = self.ndes[5](x)
        x = self.ndes[4](x)
        x = torch.cat([in_0, x], 1)
        x = self.ndes[3](x)
        x = self.ndes[2](x) 
        #x = self.bn2(x)
        x = unsqueeze(x, 2)
        x = self.ndes[1](x)
        x = self.ndes[0](x)
        #x = self.bn1(x)
        for nde in self.ndes: nde.s_span = torch.linspace(0, 1, 4)
        
        return x

    # def training_step(self, batch, batch_idx):
    #     self.iters += 1
    #     x, _ = batch
    #     x = add_uniform_noise(x)  # uniform dequantization
    #     xS = self.model(x)
    #     # prior joint likelihood p(x, ϵ)
    #     logp_x = standard_normal_logprob(xS).view(x.shape[0], -1).sum(1, keepdim=True)

model = MNISTCNF()
x = torch.randn(128, 1, 28, 28)
out = model(x)