
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

# Thanks to https://github.com/zh217/torch-dct for the original implementation of type-II DCT (1d and 2d)

import torch
from torch.fft import fft, fft2, ifft, ifft2
import torch.nn as nn
import numpy as np

def dct1d(x, norm=None, type=2):
    assert type in [2], "Only DCT type 2 is implemented."
    if type == 2:
        return _dct1d_type2(x, norm=norm)

def _dct1d_type1(x, norm=None):
    raise NotImplementedError("")

def _dct1d_type2(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1)) 

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def _dct1d_type3(x, norm=None):
    raise NotImplementedError("")

def _dct1d_type4(x, norm=None):
    raise NotImplementedError("")

def idct1d(X, norm=None, type=2):
    assert type in [2], "Only iDCT type 2 is implemented."
    if type == 2:
        return _idct1d_type2(X, norm=norm)

def _idct1d_type1(X, norm=None):
    raise NotImplementedError("")

def _idct1d_type2(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v= torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def _dct1d_type3(X, norm=None):
    raise NotImplementedError("")

def _idct1d_type4(X, norm=None):
    raise NotImplementedError("")

def dct2d(x, norm=None):
    X1 = dct1d(x, norm=norm)
    X2 = dct1d(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct2d(X, norm=None):
    x1 = idct1d(X, norm=norm)
    x2 = idct1d(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

# transforms as layers
class DiscreteFourierTransform1d(nn.Module):
    def __init__(self, norm=None, ttype=2, explicit=False):
        super(DiscreteCosineTransform1d, self).__init__()
        self.norm = norm
        self.ttype = ttype
        self.explicit = explicit
    def forward(self, x):
        return fft(x, norm=self.norm)
    def inverse(self, X):
        return ifft(X, norm=self.norm)

class DiscreteFourierTransform2d(nn.Module):
    def __init__(self, norm=None, ttype=2, explicit=False):
        super(DiscreteCosineTransform1d, self).__init__()
        self.norm = norm
        self.ttype = ttype
        self.explicit = explicit
    def forward(self, x):
        return fft2(x, norm=self.norm, type=self.ttype)
    def inverse(self, X):
        return ifft2(X, norm=self.norm)

class DiscreteCosineTransform1d(nn.Module):
    def __init__(self, norm=None, ttype=2, explicit=False):
        super(DiscreteCosineTransform1d, self).__init__()
        self.norm = norm
        self.ttype = ttype
        self.explicit = explicit
    def forward(self, x):
        return dct1d(x, norm=self.norm, type=self.ttype)
    def inverse(self, X):
        return idct1d(X, norm=self.norm)

class DiscreteCosineTransform2d(nn.Module):
    def __init__(self, norm=None, ttype=2, explicit=False):
        super(DiscreteCosineTransform1d, self).__init__()
        self.norm = norm
        self.ttype = ttype
        self.explicit = explicit
    def forward(self, x):
        return dct2d(x, norm=self.norm, ttype=self.ttype)
    def inverse(self, X):
        return idct2d(X, norm=self.norm)

