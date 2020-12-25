import torch
from torchdyn.models import *
from .utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_check_fourier():
    f = nn.Sequential(DepthCat(), GalLinear(2, 2, basisfunc=Fourier(5)))
    model = NeuralDE(f).to(device)
    x = torch.randn(128, 2).to(device)
    model(x)

def test_check_vanilla_rbf():
    f = nn.Sequential(DepthCat(), GalLinear(2, 2, basisfunc=VanillaRBF(5)))
    model = NeuralDE(f).to(device)
    x = torch.randn(128, 2).to(device)
    model(x)

def test_check_gaussian_rbf():
    f = nn.Sequential(DepthCat(), GalLinear(2, 2, basisfunc=GaussianRBF(5)))
    model = NeuralDE(f).to(device)
    x = torch.randn(128, 2).to(device)
    model(x)

def test_check_multiquad_rbf():
    f = nn.Sequential(DepthCat(), GalLinear(2, 2, basisfunc=MultiquadRBF(5)))
    model = NeuralDE(f).to(device)
    x = torch.randn(128, 2).to(device)
    model(x)

def test_check_chebychev():
    f = nn.Sequential(DepthCat(), GalLinear(2, 2, basisfunc=Chebychev(2)))
    model = NeuralDE(f).to(device)
    x = torch.randn(128, 2).to(device)
    model(x)

def test_check_polynomial():
    f = nn.Sequential(DepthCat(), GalLinear(2, 2, basisfunc=Polynomial(2)))
    model = NeuralDE(f).to(device)
    x = torch.randn(128, 2).to(device)
    model(x)