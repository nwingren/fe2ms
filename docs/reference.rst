API reference
=============

Main public interface
---------------------

The following classes make up the core of the public part of the code. These are callable directly
from the top module for convenience, but the source code is located in the ``systems`` and
``utility`` submodules.

.. autosummary::
    :toctree: generated
    :template: custom-class-template.rst

    fe2ms.ComputationVolume
    fe2ms.FEBISystemFull
    fe2ms.FEBISystemACA

The following submodules have useful contents to the classes above.

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst

    fe2ms.preconditioners
    fe2ms.materials

Internal submodules
-------------------

Submodules which are mostly internal and rarely need to be called directly.

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst
    
    fe2ms.systems
    fe2ms.result_computations
    fe2ms.assembly
    fe2ms.assembly_nonsingular_full
    fe2ms.assembly_nonsingular_aca
    fe2ms.bi_space
    fe2ms.utility