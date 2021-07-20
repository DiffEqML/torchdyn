torchdyn reference documentation
====================================
(New release 1.0 is finally out!)

torchdyn is a Python library entirely dedicated to continuous, implicit neural architectures and the numerical methods that underpin them. 

It features a self-contained numerical suite of differential equation solvers, different techniques to compute gradients of neural differential equations (continuous backsolve or interpolated adjoints, reverse-mode AD), a model zoo and an extensive set of tutorials for researchers and practitioners.

.. code:: bash

   pip install torchdyn


GitHub: `<https://github.com/diffeqml/torchdyn>`.

.. note::

    * This library is developed and maintained by **Michael Poli** & **Stefano Massaroli**, with gracious contributions from the community.
    * The work has been supported by **Prof. Jinkyoo Park** (KAIST), **Prof. Atsushi Yamashita** (The University of Tokyo) and **Prof. Hajime Asama** (The University of Tokyo).


Refer to the links below for a quickstart to core torchdyn features and a description of the library goals and design principles. 

A set of extended tutorials, covering everything from models, benchmarks and numerics can be found at `<https://github.com/diffeqml/torchdyn/notebooks>`.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/00_quickstart
   torchdyn_design
   FAQ


.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   source/torchdyn

