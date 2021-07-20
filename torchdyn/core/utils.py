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

from inspect import getfullargspec

import torch
from torchdyn.core.defunc import DEFuncBase, DEFunc
import torch.nn as nn

def standardize_vf_call_signature(vector_field, order=1, defunc_wrap=False):
    "Ensures Callables or nn.Modules passed to `ODEProblems` and `NeuralODE` have consistent `__call__` signature (t, x)"
    
    if issubclass(type(vector_field), nn.Module):
        if 't' not in getfullargspec(vector_field.forward).args:
            print("Your vector field callable (nn.Module) should have both time `t` and state `x` as arguments, "
                "we've wrapped it for you.")
            vector_field = DEFuncBase(vector_field, has_time_arg=False)
    else: 
        # argspec for lambda functions needs to be done on the function itself
        if 't' not in getfullargspec(vector_field).args:
            print("Your vector field callable (lambda) should have both time `t` and state `x` as arguments, "
                "we've wrapped it for you.")
            vector_field = DEFuncBase(vector_field, has_time_arg=False)   
        else: vector_field = DEFuncBase(vector_field, has_time_arg=True) 
    if defunc_wrap: return DEFunc(vector_field, order)
    else: return vector_field

