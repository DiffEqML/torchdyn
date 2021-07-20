### Applications and tutorials
The current version of `torchdyn` contains various quickstart examples / tutorials which explore different aspects of continuous / implicit learning and related numerical methods. Most tutorials are kept up-to-date in case of API changes of underlying libraries. We automatically validate via a quick dry run in `test/validate_tutorials.py`. These are indicated by ✅. Older tutorials ⬜️ might require minimal API changes to get working and are not automatically validated, though the goal is to eventually extend testing. Working on ensuring older tutorials are still runnable represents a perfect opportunity to get involved in the project, requiring minimal familiarity with the codebase. 

We organize the tutorials in modules. `00_quickstart.ipynb` offers a general overview of `torchdyn` features, including some comments on design philosophy and goals of the library. 

Each module is then focused on a specific aspect of continuous or implicit learning. For the moment, we offer the following modules and tutorials:

### Module 1: Neural Differential Equations
We empirically verify several properties of Neural ODEs, and develop strategies to alleviate some of their weaknesses. Augmentation, depth-variance and more are discussed here.

* ✅ `01_neural_ode_cookbook`: here, we explore the API and how to define Neural ODE variants within `torchdyn`
* ✅ `02_crossing_trajectories`: a standard benchmark problem, highlighting expressivity limitations of Neural ODEs, and how they can be addressed
* ✅ `03_augmentation_strategies`: augmentation API for Neural ODEs
* ✅ `04_higher_order`: higher-order Neural ODE variants for classification


### Module 2: Numerics and Optimization
This module is concerned with the numerics behind neural and non-neural differential equations. We provide examples of `torchdyn` numerics API, including advanced methods such as multiple shooting algorithms and hypersolvers.

* ✅ `01_hypersolver_odeint`: solve ODEs with hybridized neural networks + ODE solvers: the hypersolver API
* ✅ `02_multiple_shooting`: get familiar with `torchdyn`'s API dedicated to multiple shooting ODE solvers.
* ✅ `03_hybrid_odeint`: learn how to simulate hybrid (potentially multi-mode) dynamical systems via `odeint_hybrid`.
* ✅ `04_generalized_adjoint`: introduce integral losses in your Neural ODE training [[18](https://arxiv.org/abs/2003.08063)] to track a sinusoidal signal

### Module 3: Tasks and Benchmarks
Here, we showcase how `torchdyn` models can be used in various machine learning and control tasks. The focus is on developing the problem setting rather than applying complex models.


* ⬜️ `01_optimal_control`: direct optimal control of dynamical systems via the Neural ODE API.
* ⬜️ `02_pde_optimal_control`: fast optimal control of a Timoshenko beam via Multiple Shooting Layers and root tracking.
* ⬜️ `03_density_estimation`: density estimation with continuous normalizing flows.
* ⬜️ `04_image_classification`: convolutional Neural ODEs for digit classification on MNIST

### Module 4: Models 
This module offers an overview of several specialized continuous or implicit models. 

* ⬜️ `01_approximate_normalizing_flows`: recover densities with FFJORD variants of continuous normalizing flows [[19](https://arxiv.org/abs/1810.01367)]
* ⬜️ `02_multiple_shooting_layers`: apply multiple shooting layers to time series classification, speeding up Neural CDEs.
* ⬜️ `03_hypersolver_optimal_control`: speed up direct optimal control of ODE with hypersolvers.
* ⬜️ `04_hamiltonian_networks`: learn dynamics of energy preserving systems with a simple implementation of `Hamiltonian Neural Networks` in `torchdyn` [[10](https://arxiv.org/abs/1906.01563)]
* ⬜️ `05_lagrangian_networks`: learn dynamics of energy preserving systems with a simple implementation of `Lagrangian Neural Networks` in `torchdyn` [[12](https://arxiv.org/abs/2003.04630)]
* ✅ `06_stable_neural_odes`: learn dynamics with `Stable Neural Flows`, a generalization of HNNs [[18](https://arxiv.org/abs/2003.08063)]
* ⬜️ `07_gde_node_classification`:  first steps into the world of Neural GDEs [[9](https://arxiv.org/abs/1911.07532)], or ODEs on graphs parametrized by graph neural networks (GNN). Classification on Cora.




#### Goals

Our current goals are to extend model zoo with pretrained Neural *DE variants and equilibrium models. 