<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/images/torchdyn_full_v2.png" width="477">
</p>

<div align="center">

---

Torchdyn is a PyTorch library dedicated to **numerical deep learning**: differential equations, integral transforms, numerical methods. Maintained by [DiffEqML](https://github.com/DiffEqML).

![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg?)
![CI](https://github.com/DiffEqML/torchdyn/actions/workflows/os-coverage.yml/badge.svg)
[![Slack](https://img.shields.io/badge/slack-chat-blue.svg?logo=slack)](https://join.slack.com/t/diffeqml/shared_invite/zt-trwgahq8-zgDqFmwS2gHYX6hsRvwDvg)
[![codecov](https://codecov.io/gh/DiffEqML/torchdyn/branch/master/graph/badge.svg)](https://codecov.io/gh/DiffEqML/torchdyn)
[![Docs](https://img.shields.io/badge/docs-passing-green.svg?)](https://torchdyn.readthedocs.io/)
[![python_sup](https://img.shields.io/badge/python-3.7+-black.svg?)](https://www.python.org/downloads/release/python-370/)

</div>

### Quick Start

Torchdyn provides utilities and layers to easily construct numerical deep learning models. For example, neural differential equations:

```
from torchdyn.core import NeuralODE

# your preferred torch.nn.Module here 
f = nn.Sequential(nn.Conv2d(1, 32, 3),
                  nn.Softplus(),
                  nn.Conv2d(32, 1, 3)
          )

nde = NeuralODE(f)
```

And you have a trainable model. Feel free to combine Torchdyn classes with any PyTorch modules to build composite models. We offer additional tools to build custom neural differential equation and implicit models, including a functional API for numerical methods. There is much more in Torchdyn other than `NeuralODE` and `NeuralSDE` classes: tutorials, a functional API to a variety of GPU-compatible numerical methods and benchmarks.

Contribute to the library with your benchmark, tasks and numerical deep learning utilities! No need to reinvent the wheel :)

## Installation

**Stable** release:

`pip install torchdyn`

Alternatively, you can build a virtual dev environment for `torchdyn` with poetry, following the steps outlined in `Contributing`.

## Documentation

Check our [docs](https://torchdyn.readthedocs.io/) for more information.

## Introduction

Interest in the blend of differential equations, deep learning and dynamical systems has been reignited by recent works [[1](https://arxiv.org/abs/1806.07366),[2](https://arxiv.org/abs/2001.04385), [3](https://arxiv.org/abs/2002.08071), [4](https://arxiv.org/abs/1909.01377)]. Modern deep learning frameworks such as PyTorch, coupled with further improvements in computational resources have allowed the continuous version of neural networks, with proposals dating back to the 80s [[5](https://ieeexplore.ieee.org/abstract/document/6814892)], to finally come to life and provide a novel perspective on classical machine learning problems.

We explore how differentiable programming can unlock the effectiveness of deep learning to accelerate progress across scientific domains, including control, fluid dynamics and in general prediction of complex dynamical systems. Conversely, we focus on models powered by numerical methods and signal processing to advance the state of AI in classical domains such as vision of natural language.

<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/animations/GalNODE.gif" width="200" height="200">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/animations/cnf_diffeq.gif" width="200" height="200">
</p>

By providing a centralized, easy-to-access collection of model templates, tutorial and application notebooks, we hope to speed-up research in this area and ultimately establish neural differential equations and implicit models as an effective tool for control, system identification and general machine learning tasks.

#### Dependencies

`torchdyn` leverages modern PyTorch best practices and handles training with `pytorch-lightning` [[6](https://github.com/PyTorchLightning/pytorch-lightning)]. We build Graph Neural ODEs utilizing the Graph Neural Networks (GNNs) API of `dgl` [[7](https://www.dgl.ai/)]. For a complete list of references, check `pyproject.toml`. We offer a complete suite of ODE solvers and sensitivity methods, extending the functionality offered by `torchdiffeq` [[1](https://arxiv.org/abs/1806.07366)]. We have light dependencies on `torchsde` [[7](https://arxiv.org/abs/2001.01328)] and `torchcde` [[8](https://arxiv.org/abs/2005.08926)].

### Applications and tutorials

`torchdyn` contains a variety of self-contained quickstart examples / tutorials built for practitioners and researchers. Refer to [the tutorial readme](tutorials/README.md)

### Contribute

 `torchdyn` is designed to be a community effort: we welcome all contributions of tutorials, model variants, numerical methods and applications related to continuous and implicit deep learning. We do not have specific style requirements, though we subscribe to many of Jeremy Howard's [ideas](https://docs.fast.ai/dev/style.html).

We use `poetry` to manage requirements, virtual python environment creation, and packaging. To install `poetry`, refer to [the docs](https://python-poetry.org/docs/).
To set up your dev environment, run `poetry install`. In example, `poetry run pytest` will then run all `torchdyn` tests inside your newly created env.

`poetry` does not currently offer a way to select `torch` wheels based on desired `cuda` and `OS`, and will install a version without GPU support. For CUDA `torch` wheels,
run `poetry run poe autoinstall-torch-cuda`, that will [automatically install](https://github.com/pmeier/light-the-torch) PyTorch based on your CUDA configuration.

If you wish to run `jupyter` notebooks within your newly created poetry environments, use `poetry run ipython kernel install --user --name=torchdyn` and switch the notebook kernel.

**Choosing what to work on:** There is always [ongoing work](https://github.com/DiffEqML/torchdyn/issues) on new features, tests and tutorials. If you wish to work on additional features not currently WIP, feel free to reach out on Slack or via email. We'll be glad to discuss details.

#### Cite us

If you find Torchdyn valuable for your research or applied projects:

```
@article{politorchdyn,
  title={TorchDyn: Implicit Models and Neural Numerical Methods in PyTorch},
  author={Poli, Michael and Massaroli, Stefano and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo and Ermon, Stefano}
}
```

<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/images/torchdyn_v2.png" width="150">
</p>
<div align="center">
