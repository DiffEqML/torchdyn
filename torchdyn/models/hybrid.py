import torch
import torch.nn as nn


class HybridNeuralDE(nn.Module):
    def __init__(self, flow, jump, out, last_output=True, reverse=False):
        super().__init__()
        self.flow, self.jump, self.out = flow, jump, out
        self.reverse, self.last_output = reverse, last_output

        # determine type of `jump` func
        # jump can be of two types:
        # either take hidden and element of sequence (e.g RNNCell)
        # or h, x_t and c (LSTMCell). Custom implementation assumes call
        # signature of type (x_t, h) and .hidden_size property
        if type(jump) == nn.modules.rnn.LSTMCell:
            self.jump_func = self._jump_latent_cell
        else:
            self.jump_func = self._jump_latent

    def forward(self, x):
        h = c = self._init_latent(x)
        Y = torch.zeros(x.shape[0], *h.shape)
        if self.reverse: x_t = x_t.flip(0)
        for t, x_t in enumerate(x):
            h, c = self.jump_func(x_t, h, c)
            h = self.flow(h)
            Y[t] = h
        Y = self.out(Y)
        return Y[-1] if self.last_output else Y

    def _init_latent(self, x):
        x = x[0]
        return torch.zeros(x.shape[0], self.jump.hidden_size).to(x.device)

    def _jump_latent(self, *args):
        x_t, h, c = args[:3]
        return self.jump(x_t, h), c

    def _jump_latent_cell(self, *args):
        x_t, h, c = args[:3]
        return self.jump(x_t, (h, c))