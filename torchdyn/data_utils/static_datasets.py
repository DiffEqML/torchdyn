"""
Classification benchmark toy dataset generation
"""

import torch
import numpy as np

def randnsphere(dim, radius):
    """Uniform sampling on a sphere of `dim` and `radius`

    :param dim: dimension of the sphere
    :type dim: int
    :param radius: radius of the sphere
    :type radius: float
    """
    v = torch.randn(dim)
    inv_len = radius / torch.sqrt(torch.pow(v, 2).sum())
    return v * inv_len

def generate_concentric_spheres(n_samples=100, noise=1e-4, dim=3, inner_radius=0.5, outer_radius=1):
    """Creates a *concentric spheres* dataset of `n_samples` data points.

    :param n_samples: number of data points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each data point
    :type noise: float
    :param dim: dimension of the spheres
    :type dim: float
    :param inner_radius: radius of the inner sphere
    :type inner_radius: float
    :param outer_radius: radius of the outer sphere
    :type outer_radius: float
    """
    X = torch.zeros((n_samples, dim))
    y = torch.zeros(n_samples)
    y[:n_samples // 2] = 1
    samples = []
    for i in range(n_samples // 2):
        samples.append(randnsphere(dim, inner_radius)[None, :])
    X[:n_samples // 2] = torch.cat(samples)
    X[:n_samples // 2] += torch.zeros((n_samples // 2, dim)).normal_(0, std=noise)
    samples = []
    for i in range(n_samples // 2):
        samples.append(randnsphere(dim, outer_radius)[None, :])
    X[n_samples // 2:] = torch.cat(samples)
    X[n_samples // 2:] += torch.zeros((n_samples // 2, dim)).normal_(0, std=noise)
    return X, y

def generate_moons(n_samples=100, noise=1e-4, **kwargs):
    """Creates a *moons* dataset of `n_samples` data points.

    :param n_samples: number of data points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each data point
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

    X, y = torch.Tensor(X), torch.Tensor(y)
    return X, y

def generate_spirals(n_samples=100, noise=1e-4, **kwargs):
    """Creates a *spirals* dataset of `n_samples` data points.

    :param n_samples: number of data points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each data point
    :type noise: float
    """
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X, y = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_samples), np.ones(n_samples))))
    X, y = torch.Tensor(X), torch.Tensor(y)
    return X, y

class ToyDataset:
    """Handles the generation of classification toy datasets"""
    def generate(self, n_samples, dataset_type, **kwargs):
        """Handles the generation of classification toy datasets
        :param n_samples: number of data points in the generated dataset
        :type n_samples: int
        :param dataset_type: {'moons', 'spirals', 'spheres'}
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



