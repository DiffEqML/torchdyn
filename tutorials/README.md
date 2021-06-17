### Applications and tutorials
The current version of `torchdyn` contains the following self-contained quickstart examples / tutorials:
* `00_quickstart`: offers a quickstart guide for `torchdyn` and Neural DEs
* `01_cookbook`: here, we explore the API and how to define Neural DE variants within `torchdyn`
* `02_image_classification`: convolutional Neural DEs on MNIST
* `03_crossing_trajectories`: a standard benchmark problem, highlighting expressivity limitations of Neural DEs, and how they can be addressed
* `04_augmentation_strategies`: augmentation API for Neural DEs

and the *advanced* tutorials, including domain specific examples (control of dynamical systems, density estimation)
* `05_generalized_adjoint`: minimize integral losses with `torchdyn`'s special integral loss adjoint  [[18](https://arxiv.org/abs/2003.08063)] to track a sinusoidal signal
* `06_higher_order`: higher-order Neural ODE variants for classification
* `07a_continuous_normalizing_flows`: recover densities with continuous normalizing flows [[1](https://arxiv.org/abs/1806.07366)]
* `07b_ffjord`: recover densities with FFJORD variants of continuous normalizing flows [[19](https://arxiv.org/abs/1810.01367)]
* `08_hamiltonian_nets`: learn dynamics of energy preserving systems with a simple implementation of `Hamiltonian Neural Networks` in `torchdyn` [[10](https://arxiv.org/abs/1906.01563)]
* `09_lagrangian_nets`: learn dynamics of energy preserving systems with a simple implementation of `Lagrangian Neural Networks` in `torchdyn` [[12](https://arxiv.org/abs/2003.04630)]
* `10_stable_neural_odes`: learn dynamics of dynamical systems with a simple implementation of `Stable Neural Flows` in `torchdyn` [[18](https://arxiv.org/abs/2003.08063)]
* `11_gde_node_classification`:  first steps into the vast world of Neural GDEs [[9](https://arxiv.org/abs/1911.07532)], or ODEs on graphs parametrized by graph neural networks (GNN). Classification on Cora