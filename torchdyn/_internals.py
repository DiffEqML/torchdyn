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

"""
Conflict manager for non-compatible Neural DE variants
"""
#ALLOWED = ['autograd', 'adjoint']

class conflict: # pragma: no cover
    def __init__(self, c_type, c_settings, c_value):
        self.conflict_type = c_type
        self.conflict_settings = c_settings
        self.conflict_value = c_value
    
def NOT_ALLOWED_ARG(st:dict):
    not_allowed = []
    
    ## non supported incompatibilities ##
    conflict_type = 'not_supported'
    
    if not st['backprop_style']=='autograd' and st['type']=='stable':
        not_allowed.append(
            conflict(
                conflict_type, 
                'back-propagation style and neural ODE type',
                [st['backprop_style'], st['type']]
            )
        )
        
    ## general API misuses ##
    conflict_type = 'general'
    
    # to add more conflicts
    if st['type'] not in ['classic', 'stable']:
        not_allowed.append(
            conflict(
                conflict_type, 
                'neural ODE type',
                [st['type']]
            )
        )
    if st['backprop_style'] not in ['autograd', 'adjoint', 'integral_adjoint']:
        not_allowed.append(
            conflict(
                conflict_type, 
                'back-propagation style',
                [st['backprop_style']]
            )
        )    
        
# DEPRECATED: new s_span API
#     if st['s_start']==st['s_end']:
#         not_allowed.append(
#             conflict(
#                 conflict_type, 
#                 'initial depth and final depth',
#                 [st['s_start'], st['s_end']]
#             )
#         )

    if st['atol']<=0:
        not_allowed.append(
            conflict(
                conflict_type, 
                'solver absolute tolerance',
                [st['atol']]
            )
        )
    if st['rtol']<=0:
        not_allowed.append(
            conflict(
                conflict_type, 
                'solver relative tolerance',
                [st['rtol']]
            )
        )
    return not_allowed
    
def compat_check(settings:dict): # pragma: no cover
    conflicts = NOT_ALLOWED_ARG(settings)
    n_conflicts = len(conflicts)
    if not n_conflicts:
        return 0

    error_msg = '\n%d Errors Found\n' % n_conflicts
    count = 1
    
    error_msg += 'General Errors:\n'
    for c in conflicts:
        if c.conflict_type == 'general':
            error_msg +='%d) Incompatible '%count+c.conflict_settings+': '+str(c.conflict_value) +'\n'
            count+=1
            
    error_msg += 'Supported Functionalities Errors:\n' 
    for c in conflicts:
        if c.conflict_type == 'not_supported':
            error_msg +='%d) '%count+c.conflict_settings+': '+str(c.conflict_value) +' not yet supported in torchdyn'
            count+=1
    raise ValueError(error_msg)
    return