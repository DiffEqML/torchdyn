# torchdyn

<p align="center">
<img src="media/logo.png" width="150" height="150">
</p>
<div align="center">

## A PyTorch based library for all things **neural differential equations**. Maintained by [DiffEqML](https://github.com/DiffEqML).

![license](https://img.shields.io/badge/license-Apache-2.0-red.svg?)
![CI](https://img.shields.io/github/workflow/status/DiffEqML/torchdyn/OS-coverage?label=CI%20testing)
[![Slack](https://img.shields.io/badge/slack-chat-blue.svg?logo=slack)](https://join.slack.com/t/diffeqml/shared_invite/zt-gq3jjj5x-LuHSB4m4gc9MsnvoF1UB6A)
[![codecov](https://codecov.io/gh/DiffEqML/torchdyn/branch/master/graph/badge.svg)](https://codecov.io/gh/DiffEqML/torchdyn)
[![Docs](https://img.shields.io/badge/docs-passing-green.svg?)](https://torchdyn.readthedocs.io/)
[![python_sup](https://img.shields.io/badge/python-3.7+-black.svg?)](https://www.python.org/downloads/release/python-370/)

</div>

### Quick Start

Neural differential equations made easy:
```
from torchdyn import NeuralODE

# your preferred torch.nn.Module here 
f = nn.Sequential(nn.Conv2d(1, 32, 3),
                  nn.Softplus(),
                  nn.Conv2d(32, 1, 3)
            )

nde = NeuralODE(f)
```
And you're good to go. The `nde` object can be seamlessly combined with other deep learning models.

## Installation

**Stable** release:

`pip install torchdyn`

* NOTE: temporarily requires additional manual installation of `torchsde`: 

`pip install git+https://github.com/google-research/torchsde.git`

**Bleeding-edge** version:

`git clone https://github.com/DiffEqML/torchdyn.git`

`cd torchdyn`

`python setup.py install`

## Documentation
https://torchdyn.readthedocs.io/

## Introduction
Interest in the blend of differential equations, deep learning and dynamical systems has been reignited by recent works [[1](https://arxiv.org/abs/1806.07366),[2](https://arxiv.org/abs/2001.04385)]. Modern deep learning frameworks such as PyTorch, coupled with progressive improvements in computational resources have allowed the continuous version of neural networks, with proposals dating back to the 80s [[3](https://ieeexplore.ieee.org/abstract/document/6814892)],  to finally come to life and provide a novel perspective on classical machine learning problems (e.g. density estimation [[4](https://arxiv.org/abs/1810.01367)])

<p align="center">
<img src="media/GalNODE.gif" width="200" height="200">
<img src="media/cnf_diffeq.gif" width="200" height="200">
</p>

Since the introduction of the `torchdiffeq` library with the seminal work [[1](https://arxiv.org/abs/1806.07366)] in 2018, little effort has been expended by the PyTorch research community on an unified framework for neural differential equations. While significant progress is being made by the Julia community and SciML [[5](https://sciml.ai/2020/03/29/SciML.html)], we believe a native PyTorch version of `torchdyn` with a focus on deep learning to be a valuable asset for the research ecosystem.

Central to the `torchdyn` approach are continuous neural networks, where *width*, *depth* (or both) are taken to their infinite limit. On the optimization front, we consider continuous "data-stream" regimes and gradient flow methods, where the dataset represents a time-evolving signal processed by the neural network to adapt its parameters.

By providing a centralized, easy-to-access collection of model templates, tutorial and application notebooks, we hope to speed-up research in this area and ultimately contribute to turning neural differential equations into an effective tool for control, system identification and common machine learning tasks.

The development of `torchdyn`, sparked by the joint work of Michael Poli & Stefano Massaroli, has been supported throughout by their *almae matres*. In particular, by  **Prof. Jinkyoo Park** (KAIST), **Prof. Atsushi Yamashita** (The University of Tokyo) and **Prof. Hajime Asama** (The University of Tokyo).

`torchdyn` is maintained by the core [DiffEqML](https://github.com/DiffEqML) team, with the generous support of the deep learning community.

### Feature roadmap
The current offering of `torchdyn` is limited compared to the rich ecosystem of continuous deep learning. If you are a researcher working in this space, and particularly if one of your previous works happens to be a `WIP feature`, feel free to reach out and help us in its implementation.

* **Basics:** quickstart ✅, cookbook ✅
* **Expressivity and augmentation:** crossing trajectories ✅, augmentation ✅, higher order ✅
* **Adjoint and beyond:** generalized adjoint ✅
* **Regularization tutorials:** regularization (coming soon) ⬜️ adaptive depth ⬜️ STEER ⬜️
* **Controlled Neural DEs**: data control ✅ neural cde (coming soon) ⬜️
* **Energy models:** hamiltonian nets ✅, lagrangian nets ✅, stable models ✅
* **Image classification:** MNIST ✅, CIFAR10 (coming soon) ⬜️
* **Density estimation tutorials:** continuous normalizing flows ✅, ffjord ✅, manifold cnf ⬜️
* **Density estimation applications:** 2d density ✅, images (coming soon) ⬜️
* **Hybrid Neural DEs:** hybrid models ✅
* **Variational Neural DE tutorials:** variational neural ode ✅ variational neural sde (coming soon) ⬜️
* **Graph Neural DEs (GDEs) tutorials:** gde node classification ✅ autoregressive gde (coming soon) ⬜️

Looking for contributions of the below variants:
* **Specific variants**: ode2vae ⬜️, anodev2 ⬜️, gruode-bayes ⬜️, neural jump stochastic ⬜️, ode2ode ⬜️

#### Dependencies
`torchdyn` leverages modern PyTorch best practices and handles training with `pytorch-lightning` [[6](https://github.com/PyTorchLightning/pytorch-lightning)]. We build Graph Neural ODEs utilizing the Graph Neural Networks (GNNs) API of `dgl` [[6](https://www.dgl.ai/)].

###  Goals of `torchdyn`
Our aim with  `torchdyn` aims is to provide a unified, flexible API  to the most recent advances in continuous deep learning. Examples include neural differential equations variants, e.g.
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

### Applications and tutorials
The current version of `torchdyn` contains the following self-contained quickstart examples / tutorials (with **a lot** more to come):
* `00_quickstart`: offers a quickstart guide for `torchdyn` and Neural DEs
* `01_cookbook`: here, we explore the API and how to define Neural DE variants within `torchdyn`
* `02_image_classification`: convolutional Neural DEs on MNIST
* `03_crossing_trajectories`: a standard benchmark problem, highlighting expressivity limitations of Neural DEs, and how they can be addressed
* `04_augmentation_strategies`: augmentation API for Neural DEs

and the *advanced* tutorials
* `05_generalized_adjoint`: minimize integral losses with `torchdyn`'s special integral loss adjoint  [[18](https://arxiv.org/abs/2003.08063)] to track a sinusoidal signal
* `06_higher_order`: higher-order Neural ODE variants for classification
* `07a_continuous_normalizing_flows`: recover densities with continuous normalizing flows [[1](https://arxiv.org/abs/1806.07366)]
* `07b_ffjord`: recover densities with FFJORD variants of continuous normalizing flows [[19](https://arxiv.org/abs/1810.01367)]
* `08_hamiltonian_nets`: learn dynamics of energy preserving systems with a simple implementation of `Hamiltonian Neural Networks` in `torchdyn` [[10](https://arxiv.org/abs/1906.01563)]
* `09_lagrangian_nets`: learn dynamics of energy preserving systems with a simple implementation of `Lagrangian Neural Networks` in `torchdyn` [[12](https://arxiv.org/abs/2003.04630)]
* `10_stable_neural_odes`: learn dynamics of dynamical systems with a simple implementation of `Stable Neural Flows` in `torchdyn` [[18](https://arxiv.org/abs/2003.08063)]
* `11_gde_node_classification`:  first steps into the vast world of Neural GDEs [[9](https://arxiv.org/abs/1911.07532)], or ODEs on graphs parametrized by graph neural networks (GNN). Classification on Cora

### Features
Check our `wiki` for a full description of available features.

### Contribute
 `torchdyn` is meant to be a community effort: we welcome all contributions of tutorials, model variants, numerical methods and applications related to continuous deep learning. We do not have specific style requirements, though we subscribe to many of Jeremy Howard's [ideas](https://docs.fast.ai/dev/style.html).

**Choosing what to work on:** There is always [ongoing work](https://github.com/DiffEqML/torchdyn/issues) on new features, tests and tutorials. Contributing to any of the above is extremely valuable to us. If you wish to work on additional features not currently WIP, feel free to reach out on Slack or via email. We'll be glad to discuss details.

#### Cite us
If you find `torchdyn` valuable for your research or applied projects:
```
@article{poli2020torchdyn,
  title={TorchDyn: A Neural Differential Equations Library},
  author={Poli, Michael and Massaroli, Stefano and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo},
  journal={arXiv preprint arXiv:2009.09346},
  year={2020}
}
```
