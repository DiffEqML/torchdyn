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
from torchdyn.core import NeuralODE
from torchdyn.core.problems import ODEProblem


@pytest.mark.parametrize('problem', [ODEProblem, NeuralODE])
@pytest.mark.parametrize('sensitivity', ['adjoint', 'interpolated_adjoint'])
@pytest.mark.parametrize('solver', ['dopri5', 'tsit5'])
def test_problem_save(problem, sensitivity, solver):
    """Test load and save problems with adjoint sensitivity methods"""
    f = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1)
    )
    model_adjoint = problem(f, sensitivity=sensitivity, solver=solver)
    torch.save(model_adjoint, 'save_test_adjoint.pt')
    assert torch.load('save_test_adjoint.pt')

