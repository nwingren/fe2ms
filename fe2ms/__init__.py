"""
Code for electromagnetic computations using the
finite element-boundary integral (FE-BI) method. Primarily based on the open source packages in
FEniCSx (https://fenicsproject.org/).
"""

import logging

from fe2ms import materials, preconditioners
from .systems import FEBISystemFull, FEBISystemACA
from .utility import ComputationVolume

LOGGER = logging.getLogger('febi')
def set_log_level(level):
    LOGGER.setLevel(level)

__all__ = [
    'materials',
    'preconditioners',
    'FEBISystemFull',
    'FEBISystemACA',
    'ComputationVolume'
]
