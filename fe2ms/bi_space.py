"""
BI mesh and basis data.

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

import numba as _nb
import numpy as _np
from scipy import linalg as _linalg
import dolfinx as _dolfinx
import ufl as _ufl
import basix as _basix
from petsc4py import PETSc as _PETSc

class BIMeshData:
    """
    Class containing BI mesh and required additional data.
    """

    def __init__(self, bi_mesh, fe_space, bi_to_fe_facets, ext_facets):

        self.mesh = bi_mesh
        self.vert_coords = bi_mesh.geometry.x

        bi_mesh.topology.create_connectivity(1, 2)
        self.edge2facet = bi_mesh.topology.connectivity(1, 2).array.reshape((-1,2))
        bi_mesh.topology.create_connectivity(1, 0)
        self.edge2vert = bi_mesh.topology.connectivity(1, 0).array.reshape((-1,2))
        bi_mesh.topology.create_connectivity(2, 0)
        self.facet2vert = bi_mesh.topology.connectivity(2, 0).array.reshape((-1,3))
        bi_mesh.topology.create_connectivity(2,1)
        self.facet2edge = bi_mesh.topology.connectivity(2,1).array.reshape((-1,3))

        self.facet_normals, self.facet_areas, self.facet_flips = _get_boundary_facet_info(
            fe_space, self.vert_coords, self.edge2vert, self.facet2edge, self.facet2vert,
            bi_to_fe_facets, ext_facets
        )

        bi_mesh.topology.create_entity_permutations()
        self.perms = bi_mesh.topology.get_cell_permutation_info()

        self.edge_facet_signs = _get_facet_signs(self)

        self.edge_lengths = _linalg.norm(
            _np.diff(self.vert_coords[self.edge2vert], axis=1)[:,0,:], axis=1
        )


class BIBasisData:
    """
    Class containing data defining basis functions on a given mesh and quadrature scheme
    """

    def __init__(self, meshdata, quad_type=_basix.QuadratureType.Default, quad_order=2):

        ref_quad_points, quad_weights = _basix.make_quadrature(
            quad_type, _basix.CellType.triangle, quad_order
        )
        element = _basix.create_element(
            _basix.ElementFamily.RT, _basix.CellType.triangle, 1
        )
        basis_uv = element.tabulate(0, ref_quad_points)[0]

        self.basis = _np.zeros(
            (*meshdata.edge2facet.shape, *quad_weights.shape, 3), dtype=_np.float64
        )
        self.divs = _np.zeros(meshdata.edge2facet.shape, dtype=_np.float64)
        _compute_basis_phys(
            basis_uv, meshdata.facet2edge, meshdata.facet2vert, meshdata.edge2facet,
            meshdata.vert_coords, meshdata.facet_areas, meshdata.edge_facet_signs,
            self.basis, self.divs
        )

        self.quad_points = _np.zeros(
            (*meshdata.facet_areas.shape, *quad_weights.shape, 3), dtype=_np.float64
        )
        _transform_quad_points(
            ref_quad_points, meshdata.facet2vert, meshdata.vert_coords, self.quad_points
        )

        self.quad_weights = quad_weights


def _get_boundary_facet_info(
    fe_space, bi_vert_coords, bi_edge2vert, bi_facet2edge, bi_facet2vert,
    bi_to_fe_facets, ext_facets
):

    # Lengthy code to project facet normals into a simple CG vector function space. These normals
    # are fairly good, but normals computed using edge cross products are more accurate.
    # Since facet areas need to be computed anyway, the normals from projections are only used to
    # correct the sign of the geometrically computed normals.
    # Projections are based on code from dolfinx_mpc, see
    # https://github.com/jorgensd/dolfinx_mpc/blob/main/python/dolfinx_mpc/utils/mpc_utils.py

    n = _ufl.FacetNormal(fe_space.mesh)
    V = _dolfinx.fem.VectorFunctionSpace(fe_space.mesh, ("CG", 1))
    nh = _dolfinx.fem.Function(V)
    u, v = _ufl.TrialFunction(V), _ufl.TestFunction(V)
    bilinear_form = _dolfinx.fem.form(_ufl.inner(u, v) * _ufl.ds)

    # Sparsity pattern with special care taken to preserve the diagonal
    all_blocks = _np.arange(V.dofmap.index_map.size_local, dtype=_np.int32)
    top_blocks = _dolfinx.fem.locate_dofs_topological(V, 2, ext_facets)
    deac_blocks = all_blocks[_np.isin(all_blocks, top_blocks, invert=True)]
    pattern = _dolfinx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()

    u_0 = _dolfinx.fem.Function(V)
    u_0.vector.set(0)

    bc_deac = _dolfinx.fem.dirichletbc(u_0, deac_blocks)
    A = _dolfinx.cpp.la.petsc.create_matrix(fe_space.mesh.comm, pattern)
    A.zeroEntries()

    form_coeffs = _dolfinx.cpp.fem.pack_coefficients(bilinear_form)
    form_consts = _dolfinx.cpp.fem.pack_constants(bilinear_form)
    _dolfinx.cpp.fem.petsc.assemble_matrix(A, bilinear_form, form_consts, form_coeffs, [bc_deac])
    A.assemblyBegin(_PETSc.Mat.AssemblyType.FLUSH) # pylint: disable=no-member
    A.assemblyEnd(_PETSc.Mat.AssemblyType.FLUSH) # pylint: disable=no-member
    _dolfinx.cpp.fem.petsc.insert_diagonal(A, bilinear_form.function_spaces[0], [bc_deac], 1.0)
    A.assemble()
    linear_form = _dolfinx.fem.form(_ufl.inner(n, v) * _ufl.ds)
    b = _dolfinx.fem.petsc.assemble_vector(linear_form)
    _dolfinx.fem.petsc.set_bc(b, [bc_deac])

    # Solve for facet normals
    solver = _PETSc.KSP().create(fe_space.mesh.comm) # pylint: disable=no-member
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)
    solver.solve(b, nh.vector)

    # Compute normals and facet areas from edges
    e0 = _np.diff(bi_vert_coords[bi_edge2vert[bi_facet2edge[:,0]]], axis=1).squeeze()
    e1 = _np.diff(bi_vert_coords[bi_edge2vert[bi_facet2edge[:,1]]], axis=1).squeeze()
    normals = _np.cross(e0, e1)
    areas = _linalg.norm(normals, axis=1, keepdims=True)
    normals /= areas
    areas = areas.squeeze() / 2

    # Correct signs of normals
    fe_facet2cell = fe_space.mesh.topology.connectivity(2,3)
    boundary_cells = fe_facet2cell.array[fe_facet2cell.offsets[bi_to_fe_facets]]
    facet_centroids = _np.mean(bi_vert_coords[bi_facet2vert], axis=1)
    normal_dirs = nh.eval(facet_centroids, boundary_cells).real
    flips = _np.sign(
        _np.sum(normals * normal_dirs, axis=1, keepdims=True)
    )
    normals *= flips

    return normals, areas, flips.ravel()


def _get_facet_signs(bi_meshdata):

    # Get signs for facets connected to all edges. For an edge i, facets 0 and 1 follow the ordering
    # in bi_meshdata.edge2facet

    edge_facet_signs = _np.zeros(bi_meshdata.edge2facet.shape, dtype=_np.int32)

    for edge in range(bi_meshdata.edge2facet.shape[0]):
        for f_index, facet in enumerate(bi_meshdata.edge2facet[edge]):

            basis_fun_index = _np.where(bi_meshdata.facet2edge[facet] == edge)[0][0]

            # Facet sign from reference element https://defelement.com/elements/raviart-thomas.html
            if basis_fun_index == 1:
                sign = 1
            else:
                sign = -1

            # Edge reflections reverse the direction of basis functions
            edge_flipped = _np.binary_repr(bi_meshdata.perms[facet], 3)[2-basis_fun_index]
            if edge_flipped == '1':
                sign *= -1

            # Permutations can sometimes cause the same sign to result for both facets. In that case
            # a sign swap must be done arbitrarily for proper RWG functions to result.
            if sign in edge_facet_signs[edge]:
                sign *= -1

            edge_facet_signs[edge, f_index] = sign

    return edge_facet_signs


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def _compute_basis_phys(
    basis_uv, facet2edge, facet2vert, edge2facet, vert_coords, facet_areas, edge_facet_signs,
    rwg_phys, rwg_divs
):

    for edge in range(rwg_phys.shape[0]):

        for i_facet in range(2):

            edge_local_index = _np.where(facet2edge[edge2facet[edge, i_facet]] == edge)[0][0]

            # If basis sign and index are not as defined for reference element
            # an edge reflection should be applied by flipping basis function
            single_rt = basis_uv.copy()[:, edge_local_index, :]
            if edge_facet_signs[edge, i_facet] == 1:
                if edge_local_index != 1:
                    single_rt *= -1
            else:
                if edge_local_index == 1:
                    single_rt *= -1

            # Vector transformation
            u_vector = vert_coords[facet2vert[edge2facet[edge, i_facet], 1]] \
                - vert_coords[facet2vert[edge2facet[edge, i_facet], 0]]
            v_vector = vert_coords[facet2vert[edge2facet[edge, i_facet], 2]] \
                - vert_coords[facet2vert[edge2facet[edge, i_facet], 0]]
            rwg_phys[edge, i_facet] += single_rt[:, :1] * u_vector + single_rt[:, 1:] * v_vector

            # Scale with the factor 2A to transform from reference element
            # Basis is kept unscaled by edge lengths for conditioning reasons
            rwg_phys[edge, i_facet] /= 2 * facet_areas[edge2facet[edge, i_facet]]
            rwg_divs[edge, i_facet] += \
                edge_facet_signs[edge, i_facet] / facet_areas[edge2facet[edge, i_facet]]


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def _transform_quad_points(
    ref_points, facet2vert, vert_coords, quad_phys
):
    for facet in range(quad_phys.shape[0]):

        for i in range(ref_points.shape[0]):
            quad_phys[facet, i] += \
                vert_coords[facet2vert[facet, 0]] * (1 - ref_points[i, 0] - ref_points[i, 1]) \
                    + vert_coords[facet2vert[facet, 1]] * ref_points[i, 0] \
                        + vert_coords[facet2vert[facet, 2]] * ref_points[i, 1]
