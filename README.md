# FE2MS

FE2MS (Fast and Efficient ElectroMagnetic Solvers) is a Python package implementing the finite element-boundary integral (FE-BI) method for electromagnetic scattering problems.

It uses many other open source packages. Some of the more specialized are [FEniCSx](https://fenicsproject.org/), [DEMCEM](https://github.com/thanospol/DEMCEM), [gmsh](https://gmsh.info/) and [AdaptOctree](https://github.com/Excalibur-SLE/AdaptOctree). Other, more general purpose packages are [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [UMFPACK](https://scikit-umfpack.github.io/scikit-umfpack/) and [Numba](https://numba.pydata.org/). For a complete list, see [environment.yml](environment.yml) and [setup.py](setup.py).

Documentation is available on [https://fe2ms.readthedocs.io](https://fe2ms.readthedocs.io).


## Installation

FE2MS is primarily based on FEniCSx which is available on macOS and Linux. However, installation of this package has only been tested on Ubuntu and the installation instructions are written for this. For Windows users, Linux can be run easily using Windows Subsystem for Linux (WSL). Installation instructions and more information can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install).

The simplest way to install the package is using mamba or conda as a package manager. Mamba is recommended due to better performance than conda.

### Install mamba

Please follow [these](https://github.com/conda-forge/miniforge#mambaforge) instructions to install mamba. Following this, it is highly recommended that you create a new environment as follows (```ENV_NAME``` can be changed to your preferred environment name). Note that the Python version 3.12 is explicitly used.

```bash
mamba create --name ENV_NAME python=3.12
mamba activate ENV_NAME
```

The use of an isolated mamba/conda environment like this is particularly important due to the mixed use of conda and pip sources for dependencies.

### Install FE2MS

The FE2MS package and its dependencies are automatically installed by
```bash
pip install https://github.com/nwingren/fe2ms/archive/refs/tags/v0.2.0.tar.gz
```

This installs the packages listed in [environment.yml](environment.yml), as well as [AdaptOctree](https://github.com/Excalibur-SLE/AdaptOctree) and [demcem4py](https://github.com/nwingren/demcem4py).

## Demos

There are two demos supplied in the ```demos``` directory of this repository. All of them are possible to run without additional dependencies. Note that the first time the code is called, the run time will be longer due to just-in-time compilation.

The first demo [coated_demo.py](demos/coated_demo.py) computes the bistatic RCS for a PEC sphere coated by two layers (air and PLA plastic). It uses a mesh [coated.msh](demos/coated.msh) which can be generated fully using gmsh in [create_mesh_coated.py](demos/create_mesh_coated.py).

The second demo [windturbine_demo.py](demos/windturbine_demo.py) computes the monostatic RCS for a wind turbine rotor constructed as a fiberglass shell with an internal structure of fiberglass spars and air. It uses the mesh [rotor.msh](demos/rotor.msh) which was generated in FreeCAD and gmsh. Note that this demo is more time and resource heavy to run than that of the coated sphere as the problem is larger and has multiple right-hand sides.

## License

Copyright (C) 2023 Niklas Wingren

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
