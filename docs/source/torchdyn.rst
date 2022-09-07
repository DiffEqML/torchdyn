NeuralODE API Quickstart
===============================
A `NeuralODE` represents the framework necessary for solving Ordinary Differential Equations (ODE) numerically.
This is typically the first access point for using `torchdyn`.

This class can be used as differentiable modules within a larger differentiable numerical program that requires
the solution of ODEs / SDEs.

.. autoclass:: torchdyn.core.neuralde.NeuralODE
   :members:

Lower-level constructs and their APIs
-----------

.. toctree::
   :maxdepth: 4

   torchdyn.core
   torchdyn.datasets
   torchdyn.models
   torchdyn.nn
   torchdyn.numerics