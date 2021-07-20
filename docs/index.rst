torchdyn reference documentation
====================================
(New release 1.0 is finally out!)

torchdyn is a Python library entirely dedicated to continuous, implicit neural architectures and the numerical methods that underpin them. 

It features a self-contained numerical suite of differential equation solvers, different techniques to compute gradients of neural differential equations (continuous backsolve or interpolated adjoints, reverse-mode AD), a model zoo and an extensive set of tutorials for researchers and practitioners.

.. code:: bash

   pip install torchdyn


GitHub page `<https://github.com/diffeqml/torchdyn.git>`.

.. note::

    * This library is developed and maintained by **Michael Poli** & **Stefano Massaroli**, with gracious contributions from the community.
    * The work has been supported by **Prof. Jinkyoo Park** (KAIST), **Prof. Atsushi Yamashita** (The University of Tokyo) and **Prof. Hajime Asama** (The University of Tokyo).

..toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/00_quickstart


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   tutorials/01_neural_ode_cookbook
   tutorials/02_classification
   tutorials/03_crossing_trajectories
   tutorials/04_augmentation_strategies
   tutorials/05_generalized_adjoint
   tutorials/06_higher_order
   tutorials/07a_continuous_normalizing_flows
   tutorials/07b_ffjord
   tutorials/08_hamiltonian_nets
   tutorials/09_lagrangian_nets
   tutorials/10_stable_neural_odes
   tutorials/11_gde_node_classification


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   source/torchdyn

