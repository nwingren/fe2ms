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
Contains electromagnetic properties of various materials on the format
(epsr, mur) where each property can be complex.
"""

VACUUM = (1., 1.)
"""Vacuum"""

# Materials used for iced reflector @ 3 GHz
ICE = (3.18 - 0.0015j, 1.)
"""
Impure clear ice @ 3.7 GHz
"""

SNOW = (1.6 - 0.000288j, 1.)
"""
Wind-packed snow @ 3GHz
"""

RIME = (2.4 - 0.000846j, 1.)
"""
Hard rime @ 3 GHz
"""

# From master thesis (divinycell HT)
FOAM = (3., 1.)
"""
Divinycell HT foam core
"""

# From ETEN10 lab manual
FR4 = (4.4 - 0.11j, 1.)
"""
FR-4 @ 2.441 GHz (?)
"""

FR4_1MHz = (4.9 - 0.086j, 1.)
"""
FR-4 55 % resin by volume @ 1 MHz from https://doi.org/10.1007/BF02657420
"""

# Dry sand (from https://doi.org/10.1109/36.655342)
SAND = (2.5 - 0.05j, 1.)
"""
Dry sand @ 1 GHz from https://doi.org/10.1109/36.655342
"""

# 3D printed materials @ 1 GHz (from https://doi.org/10.1109/COMITE.2019.8733590)
ABS = (2.6 - 0.03j, 1.)
"""
3D printed ABS @ 1 GHz from https://doi.org/10.1109/COMITE.2019.8733590
"""

PET = (2.9 - 0.045j, 1.)
"""
3D printed PET @ 1 GHz from https://doi.org/10.1109/COMITE.2019.8733590
"""

PLA = (2.7 - 0.01j, 1.)
"""
3D printed PLA @ 1 GHz from https://doi.org/10.1109/COMITE.2019.8733590
"""

PLA_60GHz = (2.84 - 0.043j, 1.)
"""
3D printed PLA @ 60 GHz, mean of anisotropic values from https://doi.org/10.1109/ICECom.2016.7843900
"""
