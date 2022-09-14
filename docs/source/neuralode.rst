NeuralODE
==========
A `NeuralODE` represents the framework necessary for solving Ordinary Differential Equations (ODE) numerically.
This is typically the first access point for using `torchdyn`.

This class can be used when differentiable modules that solve ODEs are needed within a larger differentiable
numerical program

.. autoclass:: torchdyn.core.neuralde.NeuralODE
   :members:


At a high level, it is clear that calling `NeuralODE` integrates the parameterized Differential Equation over the
specified time-span. (The time-axis can be any other dimension. In pure ML applications, it's often an application
specific virtual dimension)

At a middle level, `NeuralODE`, upon instantiation, wraps the provided `vector_field` i.e. the NN that parameterizes the
differential equation in a class called `DEFunc`.

At a lower level, a `NeuralODE` inherits from `ODEProblem`. The `ODEProblem` is the object that ties together the
vector field (e.g. an NN), the solver (e.g. RK4), and the sensitivity algorithm (e.g. adjoint) into a single construct.
The `ODEProblem` requires a `DEFunc`. For this reason, the vector field is wrapped in a `DEFunc`.