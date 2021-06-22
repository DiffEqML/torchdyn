###  Goals of `torchdyn`
Our aim with  `torchdyn` aims is to provide a unified, flexible API to aid in the implementation of recent advances in continuous and implicit learning. Some models already implemented, either here under `torchdyn.models` or in the tutorials, are:

* Neural Ordinary Differential Equations (Neural ODE) [[1](https://arxiv.org/abs/1806.07366)]
* Galerkin Neural ODE [[2](https://arxiv.org/abs/2002.08071)]
* Neural Stochastic Differential Equations (Neural SDE) [[3](https://arxiv.org/abs/1905.09883),[4](https://arxiv.org/abs/1906.02355)]
* Graph Neural ODEs [[5](https://arxiv.org/abs/1911.07532)]
* Hamiltonian Neural Networks [[6](https://arxiv.org/abs/1906.01563)]

Recurrent or "hybrid" versions for sequences
* ODE-RNN [[7](https://arxiv.org/abs/1907.03907)]

Neural numerical methods
* Hypersolvers [[12](https://arxiv.org/pdf/2007.09601.pdf)]

Augmentation strategies to relieve neural differential equations of their expressivity limitations and reduce the computational burden of the numerical solver
* ANODE (0-augmentation) [[8](https://arxiv.org/abs/1904.01681)]
* Input-layer augmentation [[9](https://arxiv.org/abs/2002.08071)]
* Higher-order augmentation [[10](https://arxiv.org/abs/2002.08071)]

Various sensitivity algorithms / variants
* Integral loss adjoint [[11](https://arxiv.org/abs/2003.08063)]