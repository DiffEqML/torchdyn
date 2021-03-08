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

"""
Commonly used static datasets. Several can be used in both `density estimation` as well as classification
"""

import math

import numpy as np
import torch
from torch import sqrt, pow, cat, zeros, Tensor
from scipy.integrate import solve_ivp
from torchdyn import TTuple, Tuple
from sklearn.neighbors import KernelDensity
from torch.distributions import Normal


def randnsphere(dim:int, radius:float) -> Tensor:
    """Uniform sampling on a sphere of `dim` and `radius`

    :param dim: dimension of the sphere
    :type dim: int
    :param radius: radius of the sphere
    :type radius: float
    """
    v = torch.randn(dim)
    inv_len = radius / sqrt(pow(v, 2).sum())
    return v * inv_len


def generate_concentric_spheres(n_samples:int=100, noise:float=1e-4, dim:int=3,
                                inner_radius:float=0.5, outer_radius:int=1) -> TTuple:
    """Creates a *concentric spheres* dataset of `n_samples` datasets points.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    :param dim: dimension of the spheres
    :type dim: float
    :param inner_radius: radius of the inner sphere
    :type inner_radius: float
    :param outer_radius: radius of the outer sphere
    :type outer_radius: float
    """
    X, y = zeros((n_samples, dim)), torch.zeros(n_samples)
    y[:n_samples // 2] = 1
    samples = []
    for i in range(n_samples // 2):
        samples.append(randnsphere(dim, inner_radius)[None, :])
    X[:n_samples // 2] = cat(samples)
    X[:n_samples // 2] += zeros((n_samples // 2, dim)).normal_(0, std=noise)
    samples = []
    for i in range(n_samples // 2):
        samples.append(randnsphere(dim, outer_radius)[None, :])
    X[n_samples // 2:] = cat(samples)
    X[n_samples // 2:] += zeros((n_samples // 2, dim)).normal_(0, std=noise)
    return X, y


def generate_moons(n_samples:int=100, noise:float=1e-4, **kwargs) -> TTuple:
    """Creates a *moons* dataset of `n_samples` datasets points.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if noise is not None:
        X += np.random.rand(n_samples, 1) * noise

    X, y = Tensor(X), Tensor(y).long()
    return X, y


def generate_spirals(n_samples=100, noise=1e-4, **kwargs) -> TTuple:
    """Creates a *spirals* dataset of `n_samples` datasets points.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X, y = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_samples), np.ones(n_samples))))
    X, y = torch.Tensor(X), torch.Tensor(y).long()
    return X, y


def generate_gaussians(n_samples=100, n_gaussians=7, dim=2,
                       radius=0.5, std_gaussians=0.1, noise=1e-3) -> TTuple:
    """Creates `dim`-dimensional `n_gaussians` on a ring of radius `radius`.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param n_gaussians: number of gaussians distributions placed on the circle of radius `radius`
    :type n_gaussians: int
    :param dim: dimension of the dataset. The distributions are placed on the hyperplane (x1, x2, 0, 0..) if dim > 2
    :type dim: int
    :param radius: radius of the circle on which the distributions lie
    :type radius: int
    :param std_gaussians: standard deviation of the gaussians.
    :type std_gaussians: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    X = torch.zeros(n_samples * n_gaussians, dim) ; y = torch.zeros(n_samples * n_gaussians).long()
    angle = torch.zeros(1)
    if dim > 2: loc = torch.cat([radius*torch.cos(angle), radius*torch.sin(angle), torch.zeros(dim-2)])
    else: loc = torch.cat([radius*torch.cos(angle), radius*torch.sin(angle)])
    dist = Normal(loc, scale=std_gaussians)

    for i in range(n_gaussians):
        angle += 2*math.pi / n_gaussians
        if dim > 2: dist.loc = torch.Tensor([radius*torch.cos(angle), torch.sin(angle), radius*torch.zeros(dim-2)])
        else: dist.loc = torch.Tensor([radius*torch.cos(angle), radius*torch.sin(angle)])
        X[i*n_samples:(i+1)*n_samples] = dist.sample(sample_shape=(n_samples,)) + torch.randn(dim)*noise
        y[i*n_samples:(i+1)*n_samples] = i
    return X, y


def generate_gaussians_spiral(n_samples=100, n_gaussians=7, n_gaussians_per_loop=4, dim=2,
                              radius_start=1, radius_end=0.2, std_gaussians_start=0.3,
                              std_gaussians_end=0.1, noise=1e-3) -> TTuple:
    """Creates `dim`-dimensional `n_gaussians` on a spiral.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param n_gaussians: number of total gaussians distributions placed on the spirals
    :type n_gaussians: int
    :param n_gaussians_per_loop: number of gaussians distributions per loop of the spiral
    :type n_gaussians_per_loop: int
    :param dim: dimension of the dataset. The distributions are placed on the hyperplane (x1, x2, 0, 0..) if dim > 2
    :type dim: int
    :param radius_start: starting radius of the spiral
    :type radius_start: int
    :param radius_end: end radius of the spiral
    :type radius_end: int
    :param std_gaussians_start: standard deviation of the gaussians at the start of the spiral. Linear interpolation (start, end, num_gaussians)
    :type std_gaussians_start: int
    :param std_gaussians_end: standard deviation of the gaussians at the end of the spiral
    :type std_gaussians_end: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    X = torch.zeros(n_samples * n_gaussians, dim) ; y = torch.zeros(n_samples * n_gaussians).long()
    angle = torch.zeros(1)
    radiuses = torch.linspace(radius_start, radius_end, n_gaussians)
    std_devs = torch.linspace(std_gaussians_start, std_gaussians_end, n_gaussians)

    if dim > 2: loc = torch.cat([radiuses[0]*torch.cos(angle), radiuses[0]*torch.sin(angle), torch.zeros(dim-2)])
    else: loc = torch.cat([radiuses[0]*torch.cos(angle), radiuses[0]*torch.sin(angle)])
    dist = Normal(loc, scale=std_devs[0])

    for i in range(n_gaussians):
        angle += 2*math.pi / n_gaussians_per_loop
        if dim > 2: dist.loc = torch.Tensor([radiuses[i]*torch.cos(angle), torch.sin(angle), radiuses[i]*torch.zeros(dim-2)])
        else: dist.loc = torch.Tensor([radiuses[i]*torch.cos(angle), radiuses[i]*torch.sin(angle)])
        dist.scale = std_devs[i]

        X[i*n_samples:(i+1)*n_samples] = dist.sample(sample_shape=(n_samples,)) + torch.randn(dim)*noise
        y[i*n_samples:(i+1)*n_samples] = i
    return X, y


def generate_diffeqml(n_samples=100, noise=1e-3) -> Tuple[Tensor, None]:
    """Samples `n_samples` 2-dim points from the DiffEqML logo.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    mu = 1
    X0 = [[0,2],[-1.6, -1.2],[1.6, -1.2],]
    ti, tf = 0., 3.2
    t = np.linspace(ti,tf,500)
    # define the ODE model
    def odefunc(t,x):
        dxdt = -x[1] + mu*x[0]*(1- x[0]**2 - x[1]**2)
        dydt =  x[0] + mu*x[1]*(1- x[0]**2 - x[1]**2)
        return np.array([dxdt,dydt]).T
    # integrate ODE
    X = []
    for x0 in X0:
        sol = solve_ivp(odefunc, [ti, tf], x0, method='LSODA', t_eval=t)
        X.append(torch.tensor(sol.y.T).float())

    theta = torch.linspace(0,2*np.pi, 1000)
    X.append(torch.cat([2*torch.cos(theta)[:,None], 2*torch.sin(theta)[:,None]],1))
    X = torch.cat(X)
    k = KernelDensity(kernel='gaussian',bandwidth=.01)
    k.fit(X)

    X = torch.tensor(k.sample(n_samples) + noise*np.random.randn(n_samples, 2)).float()
    return X, None


class ToyDataset:
    """Handles the generation of classification toy datasets"""
    def generate(self, n_samples:int, dataset_type:str, **kwargs) -> TTuple:
        """Handles the generation of classification toy datasets
        :param n_samples: number of datasets points in the generated dataset
        :type n_samples: int
        :param dataset_type: {'moons', 'spirals', 'spheres', 'gaussians', 'gaussians_spiral', diffeqml'}
        :type dataset_type: str
        :param dim: if 'spheres': dimension of the spheres
        :type dim: float
        :param inner_radius: if 'spheres': radius of the inner sphere
        :type inner_radius: float
        :param outer_radius: if 'spheres': radius of the outer sphere
        :type outer_radius: float
        """
        if dataset_type == 'moons':
            return generate_moons(n_samples=n_samples, **kwargs)
        elif dataset_type == 'spirals':
            return generate_spirals(n_samples=n_samples, **kwargs)
        elif dataset_type == 'spheres':
            return generate_concentric_spheres(n_samples=n_samples, **kwargs)
        elif dataset_type == 'gaussians':
            return generate_gaussians(n_samples=n_samples, **kwargs)
        elif dataset_type == 'gaussians_spiral':
            return generate_gaussians_spiral(n_samples=n_samples, **kwargs)
        elif dataset_type == 'diffeqml':
            return generate_diffeqml(n_samples=n_samples, **kwargs)
