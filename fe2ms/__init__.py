# Copyright (C) 2023 Niklas Wingren

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
