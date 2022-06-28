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

import pytest

import torch
from torchdyn.transforms import dct1d, idct1d, dct2d, idct2d
from scipy.fftpack import dct as scipy_dct, idct as scipy_idct, dctn as scipy_dctn, idctn as scipy_idctn

def test_dctII_scipy_1d():
    """Test accuracy of type-II DCT"""
    torch.manual_seed(1234)
    x = torch.randn(2056)
    X = dct1d(x, norm='ortho')
    y = idct1d(X, norm='ortho')

    X_scipy = scipy_dct(x.numpy(), norm='ortho', type=2)
    y_scipy = scipy_idct(X_scipy, norm='ortho', type=2)

    # tolerances are really not strict enough, but will do for now
    assert torch.allclose(X, torch.from_numpy(X_scipy), rtol=1e-4, atol=1e-6)
    assert torch.allclose(y, torch.from_numpy(y_scipy), rtol=1e-4, atol=1e-6)

def test_dctII_scipy_2d():
    """Test accuracy of type-II DCT"""
    torch.manual_seed(1234)
    x = torch.randn(128, 128)
    X = dct2d(x, norm='ortho')
    y = idct2d(X, norm='ortho')

    X_scipy = scipy_dctn(x.numpy(), norm='ortho', type=2, axes=(0, 1))
    y_scipy = scipy_idctn(X_scipy, norm='ortho', type=2, axes=(0, 1))

    # tolerances are really not strict enough, but will do for now
    assert torch.allclose(X, torch.from_numpy(X_scipy), rtol=1e-4, atol=1e-6)
    assert torch.allclose(y, torch.from_numpy(y_scipy), rtol=1e-4, atol=1e-6)


