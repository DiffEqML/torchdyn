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
import torch.utils.data as data
import pytorch_lightning as pl
from torchdyn.models import *; from torchdyn.datasets import *
from torchdyn import *

def test_adjoint_autograd():
    """Test generation of (vanilla) version of all static_datasets"""
    d = ToyDataset()
    for dataset_type in ['moons', 'spirals', 'spheres', 'gaussians', 'gaussians_spiral', 'diffeqml']:
        X, yn = d.generate(n_samples=512, noise=0.2, dataset_type=dataset_type)  
    
    