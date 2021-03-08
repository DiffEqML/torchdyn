__version__ = '0.2.0'
__author__  = 'Michael Poli, Stefano Massaroli et al.'

from torchdyn.sensitivity.adjoint import Adjoint
from torchdyn.models import NeuralODE, NeuralSDE
from torchdyn.nn.utils import DepthCat, Augmenter, DataControl

from torch import Tensor
from typing import Tuple

TTuple = Tuple[Tensor, Tensor]

__all__ = ['Adjoint', 'NeuralODE', 'NeuralSDE', 'DepthCat', 'Augmenter', 'DataControl']