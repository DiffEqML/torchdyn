import torch
import torch.nn as nn

class SqueezeLayer(nn.Module):
    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            return self._upsample(x, logpx, reg_states)
        else:
            return self._downsample(x, logpx, reg_states)

    def _downsample(self, x, logpx=None, reg_states=tuple()):
        squeeze_x = squeeze(x, self.downscale_factor)
        return squeeze_x, logpx, reg_states

    def _upsample(self, y, logpy=None, reg_states=tuple()):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        return unsqueeze_y, logpy, reg_states


