###  Goals of `torchdyn`
Our aim with  `torchdyn` aims is to provide a unified, flexible API  to the most recent advances in continuous and implicit neural networks. Examples include neural differential equations variants, e.g.

* Neural Ordinary Differential Equations (Neural ODE) [[1](https://arxiv.org/abs/1806.07366)]
* Neural Stochastic Differential Equations (Neural SDE) [[7](https://arxiv.org/abs/1905.09883),[8](https://arxiv.org/abs/1906.02355)]
* Graph Neural ODEs [[9](https://arxiv.org/abs/1911.07532)]
* Hamiltonian Neural Networks [[10](https://arxiv.org/abs/1906.01563)]

Depth-variant versions,
* ANODEv2 [[11](https://arxiv.org/abs/1906.04596)]
* Galerkin Neural ODE [[12](https://arxiv.org/abs/2002.08071)]

Recurrent or "hybrid" versions
* ODE-RNN [[13](https://arxiv.org/abs/1907.03907)]
* GRU-ODE-Bayes [[14](https://arxiv.org/abs/1905.12374)]

Augmentation strategies to relieve neural differential equations of their expressivity limitations and reduce the computational burden of the numerical solver
* ANODE (0-augmentation) [[15](https://arxiv.org/abs/1904.01681)]
* Input-layer augmentation [[16](https://arxiv.org/abs/2002.08071)]
* Higher-order augmentation [[17](https://arxiv.org/abs/2002.08071)]

Alternative or modified adjoint training techniques
* Integral loss adjoint [[18](https://arxiv.org/abs/2003.08063)]
* Checkpointed adjoint [[19](https://arxiv.org/abs/1902.10298)]