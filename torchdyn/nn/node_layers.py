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

class Augmenter(nn.Module):
    """Augmentation class. Can handle several types of augmentation strategies for Neural DEs.

    :param augment_dims: number of augmented dimensions to initialize
    :type augment_dims: int
    :param augment_idx: index of dimension to augment
    :type augment_idx: int
    :param augment_func: nn.Module applied to the input datasets of dimension `d` to determine the augmented initial condition of dimension `d + a`.
                        `a` is defined implicitly in `augment_func` e.g. augment_func=nn.Linear(2, 5) augments a 2 dimensional input with 3 additional dimensions.
    :type augment_func: nn.Module
    :param order: whether to augment before datasets [augmentation, x] or after [x, augmentation] along dimension `augment_idx`. Options: ('first', 'last')
    :type order: str
    """
    def __init__(self, augment_idx:int=1, augment_dims:int=5, augment_func=None, order='first'):
        super().__init__()
        self.augment_dims, self.augment_idx, self.augment_func = augment_dims, augment_idx, augment_func
        self.order = order

    def forward(self, x: torch.Tensor):
        if not self.augment_func:
            new_dims = list(x.shape)
            new_dims[self.augment_idx] = self.augment_dims

            # if-else check for augmentation order
            if self.order == 'first':
                x = torch.cat([torch.zeros(new_dims).to(x), x],
                              self.augment_idx)
            else:
                x = torch.cat([x, torch.zeros(new_dims).to(x)],
                              self.augment_idx)
        else:
            # if-else check for augmentation order
            if self.order == 'first':
                x = torch.cat([self.augment_func(x).to(x), x],
                              self.augment_idx)
            else:
                x = torch.cat([x, self.augment_func(x).to(x)],
                               self.augment_idx)
        return x


class DepthCat(nn.Module):
    """Depth variable `t` concatenation module. Allows for easy concatenation of `t` each call of the numerical solver, at specified nn of the DEFunc.

    :param idx_cat: index of the datasets dimension to concatenate `t` to.
    :type idx_cat: int
    """
    def __init__(self, idx_cat=1):
        super().__init__()
        self.idx_cat, self.t = idx_cat, None

    def forward(self, x):
        t_shape = list(x.shape)
        t_shape[self.idx_cat] = 1
        t = self.t * torch.ones(t_shape).to(x)
        return torch.cat([x, t], self.idx_cat).to(x)


class DataControl(nn.Module):
    """Data-control module. Allows for datasets-control inputs at arbitrary points of the DEFunc
    """
    def __init__(self):
        super().__init__()
        self._control = None

    def forward(self, x):
        return torch.cat([x, self._control], 1).to(x)