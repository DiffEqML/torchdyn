import torch
import torch.nn as nn

class LatentNeuralDE(nn.Module):
    def __init__(self, encoder, decoder, out):
        super().__init__()
        self.encoder, self.decoder, self.out = encoder, decoder, out
        
    def forward(self, x, s_span):
        z = encoder(x)
        z, qz0_mean, qz0_logvar = self.reparametrize(z)
        decoded_traj = self.decoder.trajectory(z, s_span)
        
        outs = []
        for el in decoded_traj:
            outs += [self.out(el)[None]]
        return torch.cat(outs), qz0_mean, qz0_logvar
    
    def reparametrize(self, z):
        dim = z.shape[1] // 2
        qz0_mean, qz0_logvar = z[:, :latent_dim], z[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0, qz0_mean, qz0_logvar