# FE2MS

FE2MS (Fast and Efficient ElectroMagnetic Solvers) is a Python package based on [FEniCSx](https://fenicsproject.org/) implementing the finite element-boundary integral (FE-BI) method for electromagnetic scattering problems.


## Installation

FE2MS is based on FEniCSx which is available on macOS and Linux. However, it has only been tested on Ubuntu and the installation instructions are written for this.

The package only has a minimal setup now (without automatic installation of dependencies) and these installation instruction are likely to change in the future.
Installation using mamba (similar to conda) is recommended. The instructions are as follows.

### Install mamba

Please follow [these](https://github.com/conda-forge/miniforge#mambaforge) instructions to install mamba. Following this, it is recommended that you create a new environment as follows ("ENV_NAME" can be changed to your preferred environment name).

```bash
mamba create --clone base --name ENV_NAME
mamba activate ENV_NAME
```

### Install FEniCSx

This will create a mamba environment fe2ms on your system and install fenicsx there.

```bash
mamba install fenics-dolfinx=0.6.0 mpich petsc=*=complex*
```

### Install other Python packages

This will install other Python packages into the fe2ms environment.

```bash
mamba install scipy matplotlib numba pybind11 scikit-umfpack python-gmsh
cd ~
git clone https://github.com/Excalibur-SLE/AdaptOctree.git
pip install --no-deps AdaptOctree/
```

### Install DEMCEM

This will build DEMCEM and the necessary pybind11 bindings for it.

```bash
mamba install cxx cmake

cd ~
git clone https://github.com/thanospol/DEMCEM.git
export DEMCEM_DIR=$PWD/DEMCEM

cd ~
git clone https://github.com/nwingren/fe2ms.git
cd fe2ms/fe2ms/bindings
mkdir build
cd build
cmake ../
make install
```

### Finalize installation

Finally, this will install the fe2ms package into the mamba environment.

```bash
cd ~
pip install fe2ms/
```

## License

Copyright (C) 2023 Niklas Wingern

FE2MS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Acknowledgements
This work was supported in part by the Swedish Armed Forces, in part by the Swedish Defence Materiel Administration, in part by the National Aeronautics Research Program and in part by the Swedish Governmental Agency for Innovation Systems.
