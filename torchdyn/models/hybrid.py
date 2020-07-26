import torch
import torch.nn as nn

class HybridNeuralDE(nn.Module):
    def __init__(self, jump, flow, out, hidden_size, last_output=True, reverse=False):
        super().__init__()
        self.flow, self.jump, self.out = flow, jump, out
        self.hidden_size, self.last_output = hidden_size, last_output
        self.reverse = reverse
        
    def forward(self, x):
        h = self._init_latent(x[0])
        Y = []
        if self.reverse: x_t = x_t.flip(0)
        for i, x_t in enumerate(x): 
            h = self.jump(x_t, h)
            h = self.flow(h)
            Y.append(self.out(h)[None])
        Y = torch.cat(Y)
        return Y[-1] if self.last_output else Y
        
    def _init_latent(self, x):
        return torch.zeros((x.shape[0], self.hidden_size)).to(x.device)
    
class NeuralCDE(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass