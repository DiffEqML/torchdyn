import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from torchdyn.models import *; from torchdyn.datasets import *
from torchdyn import *

def test_adjoint_autograd():
    """Test generation of (vanilla) version of all static_datasets"""
    d = ToyDataset()
    for dataset_type in ['moons', 'spirals', 'spheres', 'gaussians', 'gaussians_spiral', 'diffeqml']:
        X, yn = d.generate(n_samples=512, noise=0.2, dataset_type=dataset_type)  
    
    