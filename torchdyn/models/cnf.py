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
from typing import Union, Callable
from torch.autograd import grad


def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ

def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ

REQUIRES_NOISE = [hutch_trace]

class CNF(nn.Module):
    def __init__(self, net:nn.Module, trace_estimator:Union[Callable, None]=None, noise_dist=None, order=1):
        """Continuous Normalizing Flow

        :param net: function parametrizing the datasets vector field.
        :type net: nn.Module
        :param trace_estimator: specifies the strategy to otbain Jacobian traces. Options: (autograd_trace, hutch_trace)
        :type trace_estimator: Callable
        :param noise_dist: distribution of noise vectors sampled for stochastic trace estimators. Needs to have a `.sample` method.
        :type noise_dist: torch.distributions.Distribution
        :param order: specifies parameters of the Neural DE.
        :type order: int
        """
        super().__init__()
        self.net, self.order = net, order # order at the CNF level will be merged with DEFunc
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace;
        self.noise_dist, self.noise = noise_dist, None
        if self.trace_estimator in REQUIRES_NOISE:
            assert self.noise_dist is not None, 'This type of trace estimator requires specification of a noise distribution'

    def forward(self, x):
        with torch.set_grad_enabled(True):
            # first dimension is reserved to divergence propagation
            x_in = x[:,1:].requires_grad_(True)

            # the neural network will handle the datasets-dynamics here
            if self.order > 1: x_out = self.higher_order(x_in)
            else: x_out = self.net(x_in)

            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph

