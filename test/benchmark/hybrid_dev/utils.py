import torch
import torch.nn as nn

class ControlledSpringMassDamper(nn.Module):
    """Mass Spring Damper 1dof dynamics. Faster simulation than Eul-Lag version    
    :param m: mass
    :param k: spring coeff
    :param b: damper coeff
    :param qr: spring rest pos
    :param c: control flag on / off
    :param g: gravity constant
    :param l: length
    """
    def __init__(self, controller, m=1, k=.5, b=0.1, qr=0, c=0, g=9.81, l=1):
        super().__init__()
        self.controller = controller
        self.m, self.k, self.b = m, k, b
        self.qr, self.g, self.l = qr, g, l
        self.c = c # control yes/no [1./0.]
        
    def forward(self, x):
        with torch.set_grad_enabled(True):
            q, p = x[:,:1], x[:,1:]
            q = q.requires_grad_(True)
            # compute dynamics
            dqdt = p/self.m
            dpdt = -self.k*(q - self.qr) - self.m*self.g*self.l*torch.sin(q) - \
                    self.b*p/self.m + self.c*self.controller(x)
        return torch.cat([dqdt, dpdt], 1)
    
    def _energy_shaping(self, q):
        grad_Phi = self.Kp*(q-y_target) - m*g*l*torch.sin(q) - k*(q - qr)
        return -grad_Phi
        
    def _damping_injection(self, x):
        return -self.Kd*x[:,1:]/m 
    
    def _autonomous_energy(self, x):
        return (m*x[:,1:]**2)/2. + (k*(x[:,:1] - qr)**2)/2 \
               +m*g*l*(1 - torch.cos(x[:,:1]))
    
    def _energy(self, x):
        return (m*x[:,1:]**2)/2. + self.Kp*(x[:,:1]-y_target)**2