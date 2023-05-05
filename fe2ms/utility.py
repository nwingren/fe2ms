"""
Utility classes and internal functions.

Copyright (C) 2023 Niklas Wingren

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass as _dataclass
import numpy as _np
from scipy import sparse as _sparse

import gmsh as _gmsh
import dolfinx as _dolfinx
import ufl as _ufl
from dolfinx.io import gmshio as _gmshio
from mpi4py import MPI as _MPI

import fe2ms.bi_space as _bi_space


class ComputationVolume():
    """
    Mesh, material properties and boundary conditions for a FEBI computation
    volume.
    """


    def __init__(self, mesh_file, materials, ext_boundaries, pec_boundaries=None):
        """
        Load mesh from file and associate material and boundary data to it.

        Materials are used in the general constitutive relations as follows:
            D = eps0 * epsr * E + eps0 * eta0 * xi * H
            B = 1/c0 * zeta * E + mu0 * mur * H
        Each parameter can be either a scalar or a 3x3 tensor, depending on the material model. The
        models are
        Isotropic: (epsr, mur) where both parameters are scalars.
        Anisotropic: (epsr, mur) where at least one parameter is a 3x3 tensor.
        Bi-isotropic: (epsr, mur, xi, zeta) where all parameters are scalars.
        Bi-anisotropic: (epsr, mur, xi, zeta) where at least one parameter is a 3x3 tensor.
        Lossy materials are defined using complex parameters on the form epsr = epsr' - j*epsr''.

        Parameters
        ----------
        mesh_file : str
            Name of Gmsh mesh file in .msh file format.
        materials : dict
            Map from volume physical groups (int) to materials (tuple). The tuple can contain either
            two parameters (isotropic/anisotropic) or four parameters (bi-isotropic/bi-anisotropic).
            Parameters can be either scalars or numpy arrays of shape (3,3).
        ext_boundaries : list
            List of surface physical groups (int) of exterior boundary.
        pec_boundaries : list, optional
            List of surface physical groups (int) of pec boundary,
            by default None.
        """

        # Check material values
        self.bi_material = False
        self.anisotropy = 4 * [False]
        for mat_params in materials.values():
            if len(mat_params) == 4:
                self.bi_material = True
            if len(mat_params) not in  (2, 4):
                raise ValueError(
                    f'Valid material formats are (epsr, mur), (epsr, mur, xi, zeta), got {mat_params}'
                )
            for i, p in enumerate(mat_params):
                if not _np.isscalar(p):
                    if p.shape != (3,3):
                        raise ValueError(
                            'A material parameter can only be scalar or array of shape (3,3)'
                        )
                    self.anisotropy[i] = True

        self.materials = materials
        self.ext_boundaries = ext_boundaries
        self.pec_boundaries = pec_boundaries

        # Load mesh
        _gmsh.initialize()
        _gmsh.open(mesh_file)
        self.mesh, self.cell_tags, self.facet_tags = _gmshio.model_to_mesh(
            _gmsh.model, _MPI.COMM_SELF, 0
        )
        _gmsh.finalize()

        # Check materials map
        user_tags = set(self.materials.keys())
        mesh_tags = set(self.cell_tags.values)
        if user_tags != mesh_tags:
            print(f'test: {mesh_tags}')
            raise ValueError(f'Material physical groups {user_tags} do not '
                             f'match mesh cell tags {mesh_tags}')

        # Check boundaries map
        if self.pec_boundaries is None:
            user_tags = set(self.ext_boundaries)
        else:
            user_tags = set(self.ext_boundaries + self.pec_boundaries)
        mesh_tags = set(self.facet_tags.values)
        if user_tags != mesh_tags:
            raise ValueError(f'Boundary physical groups {user_tags} do not '
                             f'match mesh facet tags {mesh_tags}')


    def get_external_facets(self):
        """
        Get facets located at external boundary of computational volume.

        Returns
        -------
        ext_facets: ndarray
            Array containing DOLFINx indices of facets at external boundary.
        """

        ext_facets = _np.array([], dtype='int32')
        for bound in self.ext_boundaries:
            ext_facets = _np.append(
                ext_facets, self.facet_tags.indices[self.facet_tags.values == bound]
            )

        return ext_facets


    def get_pec_facets(self):
        """
        Get facets located at PEC boundaries of computational volume.

        Returns
        -------
        pec_facets: ndarray
            Array containing DOLFINx indices of facets at PEC boundaries.
        """

        if self.pec_boundaries is None:
            pec_facets =  None
        else:
            pec_facets = _np.array([], dtype='int32')
            for bound in self.pec_boundaries:
                pec_facets = _np.append(
                    pec_facets, self.facet_tags.indices[self.facet_tags.values == bound]
                )

        return pec_facets


    def get_epsr(self):
        """
        Get relative permittivity for all elements in the computational volume.

        Returns
        -------
        epsr : dolfinx.fem.Function
            DOLFINx function describing relative permittivity. Defined using a
            discontinuous Galerkin function space such that the permittivity is constant within
            mesh elements. If anisotropic materials are present, this space will be defined with a
            tensor element such that the tensor components are constant within mesh elements.
        """

        if (self.bi_material and any(self.anisotropy)) or self.anisotropy[0]:
            # Create discontinuous Galerkin tensor function
            ufl_element = _ufl.TensorElement('DG', cell=self.mesh.ufl_cell(), degree=0, shape=(3,3))
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ufl_element)
            Qsub = [Q.sub(i) for i in range(9)]
            epsr = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]

                # Since anisotropic materials are present, even isotropic materials are tensors
                if _np.isscalar(mat[0]):
                    tensor_mat = mat[0] * _np.eye(3).ravel()
                else:
                    tensor_mat = mat[0].ravel()

                # Each subspace (tensor component) is set separately
                for subspace, component in zip(Qsub, tensor_mat):
                    cell_dofs = subspace.dofmap.list.array[cells]
                    epsr.x.array[cell_dofs] = component
        else:
            # Create discontinuous Galerkin function
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ('DG', 0))
            epsr = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]
                epsr.x.array[cells] = mat[0]

        return epsr


    def get_inv_mur(self):
        """
        Get inverse of relative permeability for all elements in the computational volume.
        For isotropic materials this is simply 1/mur, but for anisotropic materials it is the
        inverse to the mur tensor.

        Returns
        -------
        inv_mur : dolfinx.fem.Function
            DOLFINx function describing relative permeability. Defined using a
            discontinuous Galerkin function space such that the permeability is constant within
            mesh elements. If anisotropic materials are present, this space will be defined with a
            tensor element such that the tensor components are constant within mesh elements.
        """
        if (self.bi_material and any(self.anisotropy)) or self.anisotropy[1]:
            # Create discontinuous Galerkin tensor function
            ufl_element = _ufl.TensorElement('DG', cell=self.mesh.ufl_cell(), degree=0, shape=(3,3))
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ufl_element)
            Qsub = [Q.sub(i) for i in range(9)]
            inv_mur = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]

                # Since anisotropic materials are present, even isotropic materials are tensors
                if _np.isscalar(mat[1]):
                    tensor_mat = 1/mat[1] * _np.eye(3).ravel()
                else:
                    tensor_mat = _np.linalg.inv(mat[1]).ravel()

                # Each subspace (tensor component) is set separately
                for subspace, component in zip(Qsub, tensor_mat):
                    cell_dofs = subspace.dofmap.list.array[cells]
                    inv_mur.x.array[cell_dofs] = component
        else:
            # Create discontinuous Galerkin function
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ('DG', 0))
            inv_mur = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]
                inv_mur.x.array[cells] = 1/mat[1]

        return inv_mur


    def get_xi(self):
        """
        Get parameter coupling H->D for all elements in the computational volume.

        Returns
        -------
        xi : dolfinx.fem.Function
            DOLFINx function describing the parameter. Defined using a
            discontinuous Galerkin function space such that xi is constant within
            mesh elements. If anisotropic materials are present, this space will be defined with a
            tensor element such that the tensor components are constant within mesh elements.
        """

        # Return zero-function if the material is not bi-isotropic/bi-anisotropic
        if not self.bi_material:
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ('DG', 0))
            return _dolfinx.fem.Function(Q)

        if any(self.anisotropy):
            # Create discontinuous Galerkin tensor function
            ufl_element = _ufl.TensorElement('DG', cell=self.mesh.ufl_cell(), degree=0, shape=(3,3))
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ufl_element)
            Qsub = [Q.sub(i) for i in range(9)]
            xi = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]

                # Since anisotropic materials are present, even isotropic materials are tensors
                if _np.isscalar(mat[2]):
                    tensor_mat = mat[2] * _np.eye(3).ravel()
                else:
                    tensor_mat = mat[2].ravel()

                # Each subspace (tensor component) is set separately
                for subspace, component in zip(Qsub, tensor_mat):
                    cell_dofs = subspace.dofmap.list.array[cells]
                    xi.x.array[cell_dofs] = component
        else:
            # Create discontinuous Galerkin function
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ('DG', 0))
            xi = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]
                xi.x.array[cells] = mat[2]

        return xi


    def get_zeta(self):
        """
        Get parameter coupling E->B for all elements in the computational volume.

        Returns
        -------
        zeta : dolfinx.fem.Function
            DOLFINx function describing the parameter. Defined using a
            discontinuous Galerkin function space such that zeta is constant within
            mesh elements. If anisotropic materials are present, this space will be defined with a
            tensor element such that the tensor components are constant within mesh elements.
        """

        # Return zero-function if the material is not bi-isotropic/bi-anisotropic
        if not self.bi_material:
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ('DG', 0))
            return _dolfinx.fem.Function(Q)

        if any(self.anisotropy):
            # Create discontinuous Galerkin tensor function
            ufl_element = _ufl.TensorElement('DG', cell=self.mesh.ufl_cell(), degree=0, shape=(3,3))
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ufl_element)
            Qsub = [Q.sub(i) for i in range(9)]
            zeta = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]

                # Since anisotropic materials are present, even isotropic materials are tensors
                if _np.isscalar(mat[3]):
                    tensor_mat = mat[3] * _np.eye(3).ravel()
                else:
                    tensor_mat = mat[3].ravel()

                # Each subspace (tensor component) is set separately
                for subspace, component in zip(Qsub, tensor_mat):
                    cell_dofs = subspace.dofmap.list.array[cells]
                    zeta.x.array[cell_dofs] = component
        else:
            # Create discontinuous Galerkin function
            Q = _dolfinx.fem.FunctionSpace(self.mesh, ('DG', 0))
            zeta = _dolfinx.fem.Function(Q)

            # Set values across cell tags in mesh
            for tag, mat in self.materials.items():
                cells = self.cell_tags.indices[self.cell_tags.values==tag]
                zeta.x.array[cells] = mat[3]

        return zeta


@_dataclass
class FEBISpaces:
    """
    Class for collecting FE-BI function space data
    """
    fe_space: _dolfinx.fem.FunctionSpace
    bi_meshdata: _bi_space.BIMeshData
    bi_basisdata: _bi_space.BIBasisData
    T_SV: _sparse.spmatrix
    T_VS: _sparse.spmatrix
    T_IV: _sparse.spmatrix
    T_VI: _sparse.spmatrix
    fe_size: int
    bi_size: int


@_dataclass
class FEBIBlocks:
    """
    Class for collecting FE-BI matrix blocks
    """
    K: _sparse.spmatrix
    B: _sparse.spmatrix
    P: _np.ndarray | _sparse.spmatrix
    Q: _np.ndarray | _sparse.spmatrix


def connect_fe_bi_spaces(fe_space, ext_facets):
    """
    Create a BI space from an existing FE space and list of external facets. The coupling is between
    lowest order Nédélec functions and RWG/Raviart-Thomas functions (unscaled by edge lengths).

    Parameters
    ----------
    fe_space : dolfinx.fem.FunctionSpace
        DOLFINx function space for the FE part.
    ext_facets : ndarray
        Array containing DOLFINx indices of facets at external boundary.

    Returns
    -------
    fe2bi_connector : scipy.sparse.csr_array
        Sparse matrix T^SI which transforms FE DoFs to BI DoFs.
    bi_meshdata : febicode.utility.BIMeshData
        Mesh and additional data for BI function space construction.
    """

    bi_mesh, fe_from_bi_facets_list, fe_from_bi_verts_list, unused_ \
        = _dolfinx.mesh.create_submesh(fe_space.mesh, 2, ext_facets)
    bi_meshdata = _bi_space.BIMeshData(bi_mesh, fe_space, fe_from_bi_facets_list, ext_facets)

    fe_space.mesh.topology.create_connectivity(1, 0)
    edge2vert_fe = fe_space.mesh.topology.connectivity(1, 0).array.reshape((-1,2))
    fe_space.mesh.topology.create_connectivity(2, 3)
    facet2cell_fe = fe_space.mesh.topology.connectivity(2, 3)

    num_edges_fe = fe_space.mesh.topology.index_map(1).size_local
    num_verts_fe = fe_space.mesh.topology.index_map(0).size_local
    num_edges_bi = bi_mesh.topology.index_map(1).size_local
    num_verts_bi = bi_mesh.topology.index_map(0).size_local

    # BI edges <- BI vertices
    v0 = _sparse.coo_array(
        (
            _np.ones(num_edges_bi),
            (_np.arange(num_edges_bi), bi_meshdata.edge2vert[:,0])
        ),
        shape=(num_edges_bi, num_verts_bi), dtype="float64"
    ).tocsr()
    v1 = _sparse.coo_array(
        (
            _np.ones(num_edges_bi),
            (_np.arange(num_edges_bi), bi_meshdata.edge2vert[:,1])
        ),
        shape=(num_edges_bi, num_verts_bi), dtype="float64"
    ).tocsr()
    bi_edges_from_bi_verts = v0 + v1

    # BI vertices <- FE vertices
    bi_verts_from_fe_verts = _sparse.coo_array(
        (
            _np.ones(num_verts_bi),
            (_np.arange(num_verts_bi), _np.array(fe_from_bi_verts_list))
        ),
        shape=(num_verts_bi, num_verts_fe), dtype="float64"
    ).tocsr()

    # FE vertices <- FE edges
    v0 = _sparse.coo_array(
        (
            _np.ones(num_edges_fe),
            (edge2vert_fe[:,0], _np.arange(num_edges_fe))
        ),
        shape=(num_verts_fe, num_edges_fe), dtype="float64"
    ).tocsr()
    v1 = _sparse.coo_array(
        (
            _np.ones(num_edges_fe),
            (edge2vert_fe[:,1], _np.arange(num_edges_fe))
        ),
        shape=(num_verts_fe, num_edges_fe), dtype="float64"
    ).tocsr()
    fe_verts_from_fe_edges = v0 + v1

    # Multiply for map BI edges <- FE edges. Truncate the results so that only connections with
    # matching vertex pairs remain. Flips between vertex pairs are ignored here
    fe2bi_connector = bi_verts_from_fe_verts @ fe_verts_from_fe_edges
    fe2bi_connector = bi_edges_from_bi_verts @ fe2bi_connector
    fe2bi_connector.data = fe2bi_connector.data // 2
    fe2bi_connector.eliminate_zeros()

    # Connect to FE DoFs with map FE edges <- FE DoFs
    dof2edge = _np.zeros(num_edges_fe, dtype=_np.int64)
    for e in range(num_edges_fe):
        d = _dolfinx.fem.locate_dofs_topological(fe_space, 1, [e])
        dof2edge[d] = e
    fe_edges_from_fe_dofs = _sparse.coo_array(
        (_np.ones(num_edges_fe, dtype=_np.int64), (dof2edge, _np.arange(num_edges_fe))),
        shape=(num_edges_fe, num_edges_fe), dtype="float64"
    ).tocsr()
    fe2bi_connector = fe2bi_connector @ fe_edges_from_fe_dofs

    # Add sign to connection matrix entries to account for basis function direction changes. Current
    # implementation requires looping over all nonzero entries.
    nonz = fe2bi_connector.nonzero()
    for bi_edge, fe_dof in zip(nonz[0], nonz[1]):

        bi_facet = bi_meshdata.edge2facet[bi_edge, 0]
        bi_sign = bi_meshdata.edge_facet_signs[bi_edge, 0]

        # Make a DOLFINx function and activate only the current FE DoF. Evaluate function in the
        # facet centroid and compute cross product between facet normal and this value.
        fe_dof_fun = _dolfinx.fem.Function(fe_space)
        coeffs = _np.zeros(num_edges_fe, dtype=_np.complex128)
        coeffs[fe_dof] = 1
        fe_dof_fun.x.array[:] = coeffs
        fe_vals = _np.zeros((1, 3), dtype=_np.complex128)
        cell = facet2cell_fe.links(fe_from_bi_facets_list[bi_facet])
        centroid = _np.mean(
            bi_meshdata.vert_coords[bi_meshdata.facet2vert[bi_facet]], axis=0, keepdims=True
        )
        fe_dof_fun.eval(centroid, cell, fe_vals)
        cross = _np.cross(bi_meshdata.facet_normals[bi_facet], fe_vals.real.ravel())

        # Find direction of RWG function in facet centroid. A comparison between this and the
        # previously computed cross product gives the sign of the connection matrix entry.
        edge_verts_bi = bi_meshdata.edge2vert[bi_edge]
        free_vert_bi = _np.setdiff1d(
            bi_meshdata.facet2vert[bi_facet], edge_verts_bi, assume_unique=True
        )[0]
        rwg_dir = bi_sign * (centroid.ravel() - bi_meshdata.vert_coords[free_vert_bi])
        fe2bi_connector[bi_edge, fe_dof] *= _np.sign(cross @ rwg_dir)

    return fe2bi_connector, bi_meshdata
