torchdyn reference documentation
====================================
(New release 1.0 is finally out!)

torchdyn is a Python library entirely dedicated to continuous, implicit neural architectures and the numerical methods that underpin them. 

It features a self-contained numerical suite of differential equation and root solvers, including sensitivity methods (continuous backsolve or interpolated adjoints, reverse-mode AD). 
The library further contains a model zoo and an extensive set of tutorials for researchers and practitioners.

.. code:: bash

   pip install torchdyn


GitHub: `<https://github.com/diffeqml/torchdyn>`.

.. note::

    * This library is developed and maintained by **Michael Poli** & **Stefano Massaroli**, with gracious contributions from the community.


Refer to the links below for a quickstart to core torchdyn features and a description of the library goals and design principles. 

A set of extended tutorials, covering everything from models, benchmarks and numerics can be found at `<https://github.com/diffeqml/torchdyn/tutorials>`.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/quickstart
   

.. toctree::
   :maxdepth: 1
   :caption: General Information

   contributing
   FAQ

.. toctree::
   :maxdepth: 1
   :caption: API documentation

   source/torchdyn

