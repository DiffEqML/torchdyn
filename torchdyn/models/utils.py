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
import warnings


SCIPY_SOLVERS = {
    "scipy_LSODA": {'method':'scipy_solver', 'options':{'solver':'LSODA'}},
    "scipy_RK45": {'method':'scipy_solver', 'options':{'solver':'RK45'}},
    "scipy_RK23": {'method':'scipy_solver', 'options':{'solver':'RK23'}},
    "scipy_DOP853": {'method':'scipy_solver', 'options':{'solver':'DOP853'}},
    "scipy_BDF": {'method':'scipy_solver', 'options':{'solver':'BDF'}},
    "scipy_Radau": {'method':'scipy_solver', 'options':{'solver':'Radau'}},
}

def check_solver_compat(solver: str, sensitivity: str):
    if solver[:5] == "scipy" and solver not in SCIPY_SOLVERS:
        available_scipy_solvers = ", ".join(SCIPY_SOLVERS.keys())
        raise KeyError("Invalid Scipy Solver specified." +
                       " Supported Scipy Solvers are: " + available_scipy_solvers)

    elif solver in SCIPY_SOLVERS:
        warnings.warn(UserWarning("CUDA is not available with SciPy solvers."))

        if sensitivity == 'autograd':
            raise ValueError("SciPy Solvers do not work with autograd." +
                             " Use adjoint sensitivity with SciPy Solvers.")

def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def make_norm(state):
    state_size = state.numel()
    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm
