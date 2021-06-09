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


SCIPY_SOLVERS = {
    "scipy_LSODA": {'method':'scipy_solver', 'options':{'solver':'LSODA'}},
    "scipy_RK45": {'method':'scipy_solver', 'options':{'solver':'RK45'}},
    "scipy_RK23": {'method':'scipy_solver', 'options':{'solver':'RK23'}},
    "scipy_DOP853": {'method':'scipy_solver', 'options':{'solver':'DOP853'}},
    "scipy_BDF": {'method':'scipy_solver', 'options':{'solver':'BDF'}},
    "scipy_Radau": {'method':'scipy_solver', 'options':{'solver':'Radau'}},
}
