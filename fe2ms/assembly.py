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
Functions for assembly of the FE and BI blocks.
"""

import numba as _nb
import numpy as _np
from scipy import sparse as _sparse
from scipy.constants import epsilon_0 as _eps0, mu_0 as _mu0
import dolfinx as _dolfinx
import ufl as _ufl
from adaptoctree import tree as _tree, morton as _morton

import fe2ms.bi_space as _bi_space
import fe2ms.assembly_nonsingular_full as _assembly_full
import fe2ms.assembly_nonsingular_aca as _assembly_aca
from fe2ms.utility import(
    ComputationVolume as _ComputationVolume,
    FEBISpaces as _FEBISpaces
)
from fe2ms.bindings import demcem_bindings as _demcem_bindings


def assemble_KB_blocks(
    k0, computation_volume: _ComputationVolume,
    function_spaces: _FEBISpaces
):
    """
    Assemble K and B blocks corresponding to finite element equation.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber of problem.
    computation_volume : utility.ComputationVolume
        FEBI computation volume defining mesh, material and boundaries.
    function_spaces : utility.FEBISpaces
        Function spaces and related structures for the problem.

    Returns
    -------
    K_matrix : scipy.sparse.csr_array
        Finite element matrix K.
    B_matrix : scipy.sparse.csr_array
        Coupling matrix B (defined for BI DoFs).
    """

    # Define PEC boundary condition
    pec_facets = computation_volume.get_pec_facets()
    if pec_facets is not None:
        pec_dofs = _dolfinx.fem.locate_dofs_topological(
            function_spaces.fe_space, 2, pec_facets
        )
        zero_fun = _dolfinx.fem.Function(function_spaces.fe_space)
        with zero_fun.vector.localForm() as loc:
            loc.set(0)
        zero_fun.x.scatter_forward()
        boundary_conditions = [_dolfinx.fem.dirichletbc(zero_fun, pec_dofs)]
    else:
        boundary_conditions = []

    # Finite element operator
    u = _ufl.TrialFunction(function_spaces.fe_space)
    v = _ufl.TestFunction(function_spaces.fe_space)

    # Generate and assemble K matrix with Dirichlet bc applied
    # Forms are different for isotropic/anisotropic and bi-isotropic/bi-anisotropic materials
    # Form for bi-materials is from https://doi.org/10.1163/1569393042954857
    epsr = computation_volume.get_epsr()
    epsr.x.scatter_forward()
    inv_mur = computation_volume.get_inv_mur()
    inv_mur.x.scatter_forward()
    if computation_volume.bi_material:
        xi = computation_volume.get_xi()
        xi.x.scatter_forward()
        zeta = computation_volume.get_zeta()
        zeta.x.scatter_forward()
        form = _dolfinx.fem.form(
            (
                _ufl.inner(inv_mur * _ufl.curl(u), _ufl.curl(v))
                - k0**2 * _ufl.inner((epsr - xi * inv_mur * zeta) * u, v)
                + 1j * k0 * _ufl.inner((inv_mur * zeta) * u, _ufl.curl(v))
                - 1j * k0 * _ufl.inner((xi * inv_mur) * _ufl.curl(u), v)
            )
            * _ufl.dx
        )
    else:
        form = _dolfinx.fem.form(
            (
                _ufl.inner(inv_mur * _ufl.curl(u), _ufl.curl(v))
                - k0**2 * _ufl.inner(epsr * u, v)
            )
            * _ufl.dx
        )
    K_matrix = _dolfinx.fem.assemble_matrix(form, bcs=boundary_conditions)

    K_matrix.finalize() # pylint: disable=no-member
    K_matrix = _sparse.csr_array((K_matrix.data, K_matrix.indices, K_matrix.indptr)) # pylint: disable=no-member

    rows, cols, B_vals = _assembly_full.assemble_B_integral(
        function_spaces.bi_basisdata.basis, function_spaces.bi_basisdata.quad_weights,
        function_spaces.bi_meshdata.facet2edge, function_spaces.bi_meshdata.edge2facet,
        function_spaces.bi_meshdata.facet_areas, function_spaces.bi_meshdata.facet_normals
    )
    B_matrix = 1j * k0 *_sparse.coo_array((B_vals, (rows, cols)), shape=2*(function_spaces.bi_size,)).tocsr()

    return K_matrix, B_matrix


def assemble_bi_blocks_full(
    k0, meshdata: _bi_space.BIMeshData, basisdata: _bi_space.BIBasisData,
    quad_order_singular
):
    """
    Assemble full BI blocks for tested K- and L-operators. This uses edge-wise assembly as opposed
    to the more traditional facet-wise assembly used in MoM.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber of problem.
    meshdata : febicode.utility.BIMeshData
        Mesh data for the BI problem.
    basisdata : febicode.rwg_rt_helpers.BIBasisData
        Basis data for the BI problem.
    quad_order_singular : int
        Quadrature order to use in DEMCEM singular integral computations.

    Returns
    -------
    Kop_matrix : ndarray
        Matrix for tested K-operator.
    Lop_matrix : ndarray
        Matrix for tested L-operator.
    K_prec : sparray
        Matrix with preconditioner entries.
    L_prec : sparray
        Matrix with preconditioner entries.
    """

    bi_size = meshdata.edge2vert.shape[0]
    Kop_matrix = _np.zeros((bi_size, bi_size), dtype=_np.complex128)
    Lop_matrix = _np.zeros((bi_size, bi_size), dtype=_np.complex128)

    K_rows, K_cols, K_vals, L_rows, L_cols, L_vals, singular_entries, K_prec, L_prec = \
        _compute_singularities_KL_operators(k0, meshdata, basisdata, quad_order_singular, True)

    _assembly_full.assemble_KL_operators(
        k0, basisdata.basis, basisdata.divs, basisdata.quad_points, basisdata.quad_weights,
        meshdata.edge2facet, meshdata.facet_areas, singular_entries, Kop_matrix, Lop_matrix
    )

    Kop_matrix += _sparse.coo_array(
        (K_vals, (K_rows, K_cols)), shape=2*(meshdata.edge2vert.shape[0],)
    ).tocsr()
    Lop_matrix += _sparse.coo_array(
        (L_vals, (L_rows, L_cols)), shape=2*(meshdata.edge2vert.shape[0],)
    ).tocsr()

    return Kop_matrix, Lop_matrix, K_prec, L_prec


def assemble_bi_aca(
    k0, meshdata: _bi_space.BIMeshData, basisdata: _bi_space.BIBasisData,
    recompress, target_points, max_points, max_level, tolerance, quad_order_singular
):
    """
    Assemble BI blocks using the adaptive cross approximation (ACA). Uses AdaptOctree to subdivide
    geometry into near and far parts (far parts are treated in a multilevel fashion). Near parts are
    assembled fully into sparse arrays, and far parts are assembled using the ACA into a series of
    outer product forms. A special object is used to handle operations using the ACA data.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber of problem.
    meshdata : febicode.utility.BIMeshData
        Mesh data for the BI problem.
    basisdata : febicode.rwg_rt_helpers.BIBasisData
        Basis data for the BI problem.
    recompress : bool
        Whether to recompress ACA blocks using QR and SVD.
    target_points : int
        Target for the mean number of points per group. 'None' will set this to the square root of
        the number of BI DoFs.
    max_points : int
        Maximum number of points per groups. If not 'None', this will override the iteration for
        obtaining mean number of points per grous.
    max_level : int
        Maximum level for octree.
    tolerance : float
        Tolerance for the ACA and recompression SVD.
    quad_order_singular : int
        Quadrature order to use in DEMCEM singular integral computations.

    Returns
    -------
    Kop_near : ndarray
        Matrix for near interactions of tested K-operator.
    Lop_near : ndarray
        Matrix for near interactions of tested L-operator.
    far_operator : febi.assembly_nonsingular_aca.MultilevelOperator
        Object for ACA part of both tested K and L operators. Contains outer product forms and
        matvec methods.
    K_prec : sparray
        Matrix with preconditioner entries.
    L_prec : sparray
        Matrix with preconditioner entries.
    """

    edge_centers = _np.mean(meshdata.vert_coords[meshdata.edge2vert], axis=1)

    # Iteration to get mean number of DoFs per group within 1% of optimal value (sqrt(N_bi))
    # Ignores empty groups, of which there are a lot (this makes it tricky to set values)
    if max_points is None:
        if target_points is None:
            target_points = _np.sqrt(meshdata.edge2vert.shape[0])
        max_points = 2 * target_points
        for i in range(10):
            unbalanced_leaves = _tree.build(edge_centers, max_level, int(max_points), 1)
            depth = _tree.find_depth(unbalanced_leaves)
            balanced_leaves = _tree.balance(unbalanced_leaves, depth)

            bounds = _morton.find_bounds(edge_centers)
            center = _morton.find_center(*bounds)
            radius = _morton.find_radius(*bounds)

            edge2key_leaves = _tree.points_to_keys(edge_centers, balanced_leaves, depth, center, radius)
            dofs_per_group = _np.array([_np.sum(edge2key_leaves == g) for g in balanced_leaves])
            mean_dofs_per_group = _np.mean(dofs_per_group[dofs_per_group != 0])

            if _np.abs(mean_dofs_per_group / target_points - 1) > 0.01:
                max_points /= mean_dofs_per_group / target_points
            else:
                break

    complete_tree = _tree.complete_tree(balanced_leaves)

    edge2key_levels = _nb.typed.List()
    for level in range(depth + 1):
        edge2key_levels.append(_morton.encode_points_smt(edge_centers, level, center, radius))

    u, x, v, w = _tree.find_interaction_lists(balanced_leaves, complete_tree, depth)

    K_rows, K_cols, K_vals, L_rows, L_cols, L_vals, singular_entries, K_prec, L_prec = \
        _compute_singularities_KL_operators(k0, meshdata, basisdata, quad_order_singular, True)

    rows, cols, Kop_vals, Lop_vals = _assembly_aca.compute_KL_operators_near_octree(
        k0, basisdata.basis, basisdata.divs, basisdata.quad_points,
        basisdata.quad_weights, meshdata.edge2facet, meshdata.facet_areas,
        singular_entries, complete_tree, u, edge2key_leaves
    )

    Kop_near = _sparse.coo_array(
        (
            _np.concatenate((K_vals, Kop_vals)),
            (_np.concatenate((K_rows, rows)), _np.concatenate((K_cols, cols)))
        ),
        shape=2*(meshdata.edge2vert.shape[0],)
    ).tocsr()
    del K_rows, K_cols, K_vals, Kop_vals
    Lop_near = _sparse.coo_array(
        (
            _np.concatenate((L_vals, Lop_vals)),
            (_np.concatenate((L_rows, rows)), _np.concatenate((L_cols, cols)))
        ),
        shape=2*(meshdata.edge2vert.shape[0],)
    ).tocsr()
    del L_rows, L_cols, L_vals, rows, cols, Lop_vals

    far_operator = _assembly_aca.compute_KL_operators_far_octree(
        k0, basisdata.basis, basisdata.divs, basisdata.quad_points,
        basisdata.quad_weights, meshdata.edge2facet, meshdata.facet_areas,
        complete_tree, x, v, w, edge2key_levels, tolerance, recompress
    )

    return Kop_near, Lop_near, far_operator, K_prec, L_prec


def assemble_rhs(
    formulation, k0, meshdata: _bi_space.BIMeshData,
    basisdata: _bi_space.BIBasisData, source_fun
):
    """
    Assemble right-hand-side for a certain incident field.

    Parameters
    ----------
    formulation : {'is-efie', 'vs-efie', 'ej', 'teth'}
        Formulation used.
    k0 : float
        Free-space wavenumber of problem.
    meshdata : febicode.utility.BIMeshData
        Mesh data for the BI problem.
    basisdata : febicode.rwg_rt_helpers.BIBasisData
        Basis data for the BI problem.
    source_fun : tuple
        Tuple containing functions giving the incident electric and magnetic fields at N points of
        shape (N, 3).

    Returns
    -------
    b_inc : ndarray
        Right-hand-side (BI part).
    """

    if formulation not in ('is-efie', 'vs-efie', 'ej', 'teth'):
        raise NotImplementedError(f'Formulation \'{formulation}\' not implemented')

    bi_size = meshdata.edge2vert.shape[0]
    b_inc = _np.zeros(bi_size, dtype=_np.complex128)
    
    if formulation == 'ej':
        b_M = _np.zeros_like(b_inc)

    if source_fun is None:
        if formulation in ('is-efie', 'vs-efie'):
            return b_inc
        elif formulation == 'ej':
            return _np.concatenate((b_inc, b_M))

    for edge_m in range(bi_size):

        for i_facet_m, facet_m in enumerate(meshdata.edge2facet[edge_m]):

            # Jacobian norm for integral m
            J_m = 2 * meshdata.facet_areas[facet_m]

            b_inc[edge_m] += _np.sum(
                basisdata.basis[edge_m, i_facet_m] * basisdata.quad_weights[:,None]
                * source_fun[0](basisdata.quad_points[facet_m])
            ) * J_m
            if formulation == 'ej':
                b_M[edge_m] += _np.sum(
                    basisdata.basis[edge_m, i_facet_m] * basisdata.quad_weights[:,None]
                    * source_fun[1](basisdata.quad_points[facet_m])
                ) * J_m
            elif formulation == 'teth':
                b_inc[edge_m] += _np.sum(
                    basisdata.basis[edge_m, i_facet_m] * basisdata.quad_weights[:,None]
                    * source_fun[1](basisdata.quad_points[facet_m])
                ) * J_m

    if formulation == 'ej':
        b_inc = -1j * k0 * _np.concatenate((b_M, b_inc * _np.sqrt(_mu0 / _eps0)))
    elif formulation == 'teth':
        b_inc *= 0.5

    return b_inc


def _compute_singularities_KL_operators(
    k0, meshdata: _bi_space.BIMeshData, basisdata: _bi_space.BIBasisData,
    N_quad, gen_preconditioner
):
    """
    Computes singular integrals for all edges using DEMCEM (https://github.com/thanospol/DEMCEM).
    The weakly singular integrals and strongly singular integrals with RWG testing functions are
    those computed (for the cases ST, EA and VA). The parts of the singular matrix entries
    which correspond to non-adjacent sub-triangles are also computed here, but using standard
    quadrature.
    
    This is implemented for the integrals of the L-operator and K-operator tested by
    Raviart-Thomas functions. The K-operator is made from a principal value term and an extracted
    singularity term for basis/testing on the same triangle.

    The basis/testing functions used in DEMCEM differ from those here by a normalization factor
    for the edge length. This is compensated for in the function.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber of problem.
    meshdata : febicode.utility.BIMeshData
        Mesh data for the BI problem.
    basisdata : febicode.rwg_rt_helpers.BIBasisData
        Basis data for the BI problem.
    N_quad : int
        Quadrature order to use in DEMCEM singular integral computations.
    gen_preconditioner : bool
        Whether to generate sparsified operators for preconditioner. Will add two outputs.

    Returns
    -------
    K_rows : ndarray
        Rows of K entries containing singular terms (strong singularity).
    K_cols : ndarray
        Columns of K entries containing singular terms (strong singularity).
    K_vals : ndarray
        Values of K entries containing singular terms (strong singularity).
    L_rows : scipy.sparse.csc_array
        Rows of L entries containing singular terms (weak singularity).
    L_cols : scipy.sparse.csc_array
        Columns of L entries containing singular terms (weak singularity).
    L_vals : scipy.sparse.csc_array
        Values of L entries containing singular terms (weak singularity).
    singular_entries : set
        Set of tuples (row,col) corresponding to the computed singular terms.
    K_prec : scipy.sparse.csc_array
        Sparsified K operator. None if gen_preconditioner == False.
    L_prec : scipy.sparse.csc_array
        Sparsified L operator. None if gen_preconditioner == False.
    """

    num_edges = meshdata.edge2facet.shape[0]
    K_rows = []
    K_cols = []
    L_rows = []
    L_cols = []
    K_vals = []
    L_vals = []
    singular_entries = _nb.typed.Dict.empty(
        key_type=_nb.core.types.UniTuple(_nb.core.types.int32, 2),
        value_type=_nb.core.types.boolean
    )

    facetpairs_done = set()

    if gen_preconditioner:
        Lp_rows = []
        Lp_cols = []
        Lp_vals = []

    # Need extra connectivity for VA
    meshdata.mesh.topology.create_connectivity(0, 2)
    vert2facet = meshdata.mesh.topology.connectivity(0, 2)

    sing_vals = _np.zeros(9, dtype=_np.complex128)

    # Compute self-terms
    for facet_P, verts_P in enumerate(meshdata.facet2vert):

        facetpairs_done.add((facet_P, facet_P))

        # Switch order of vertices if this does not correspond to ordering associated with
        # positive outward normal.
        if meshdata.facet_flips[facet_P] < 0:
            local_order = _np.array([1,0,2], dtype=_np.int32)
        else:
            local_order = _np.arange(3, dtype=_np.int32)

        edges_m = meshdata.facet2edge[facet_P][local_order]

        # Find signs of RWG corresponding to edge_m in facet_P
        signs_m = _np.empty((3,), dtype=_np.int32)
        for i, e in enumerate(edges_m):
            signs_m[i] = meshdata.edge_facet_signs[e, meshdata.edge2facet[e] == facet_P][0]

        r_coords = meshdata.vert_coords[verts_P[local_order]]
        _demcem_bindings.ws_st_rwg(
            r_coords[0], r_coords[1], r_coords[2], k0, N_quad, sing_vals
        )

        L_rows += [edges_m[i // 3] for i in range(9)]
        L_cols += [edges_m[i % 3] for i in range(9)]
        L_vals += [
            sing_vals[i] / 4 / _np.pi * signs_m[i // 3] * signs_m[i % 3]
            / meshdata.edge_lengths[edges_m[i // 3]] / meshdata.edge_lengths[edges_m[i % 3]]
            for i in range(9)
        ]
        if gen_preconditioner:
            Lp_rows += L_rows[-9:]
            Lp_cols += L_cols[-9:]
            Lp_vals += L_vals[-9:]

        for i in range(9):
            singular_entries[(edges_m[i // 3], edges_m[i % 3])] = True

    # Compute edge adjacent terms
    for edge, facets in enumerate(meshdata.edge2facet):

        for facet_P in facets:

            facet_Q = facets[facets != facet_P][0]

            # Each facet pair is only done once due to symmetry
            if (max(facet_P, facet_Q), min(facet_P, facet_Q)) in facetpairs_done:
                continue

            facetpairs_done.add((max(facet_P, facet_Q), min(facet_P, facet_Q)))

            # Vertices attached to edge, but not in the correct order
            vert_1, vert_2 = meshdata.edge2vert[edge]

            # Find free vertices and get the correct order (local to facet P) for edge vertices
            vert_3 = meshdata.facet2vert[facet_P][_np.logical_and(
                meshdata.facet2vert[facet_P] != vert_1,
                meshdata.facet2vert[facet_P] != vert_2
            )][0]
            vert_4 = meshdata.facet2vert[facet_Q][_np.logical_and(
                meshdata.facet2vert[facet_Q] != vert_1,
                meshdata.facet2vert[facet_Q] != vert_2
            )][0]
            loc_free = _np.nonzero(meshdata.facet2vert[facet_P] == vert_3)[0][0]
            vert_1 = meshdata.facet2vert[facet_P][(loc_free + 1) % 3]
            vert_2 = meshdata.facet2vert[facet_P][(loc_free + 2) % 3]

            # Switch order of vertices attached to edge if this does not correspond to ordering
            # associated with positive outward normal.
            if meshdata.facet_flips[facet_P] < 0:
                vert_1, vert_2 = vert_2, vert_1

            # Local indices of vertices in P facet
            local_P = _np.empty(3, dtype=_np.int32)
            local_P[0] = _np.nonzero(meshdata.facet2vert[facet_P] == vert_1)[0][0]
            local_P[1] = _np.nonzero(meshdata.facet2vert[facet_P] == vert_2)[0][0]
            local_P[2] = _np.nonzero(meshdata.facet2vert[facet_P] == vert_3)[0][0]

            # Find local indices of vertices in Q facet after correcting order
            local_Q = _np.empty(3, dtype=_np.int32)
            local_Q[0] = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_2)[0][0]
            local_Q[1] = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_1)[0][0]
            local_Q[2] = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_4)[0][0]

            # Edges reordered according to DEMCEM local indexation
            edges_m = meshdata.facet2edge[facet_P][local_P]
            edges_n = meshdata.facet2edge[facet_Q][local_Q]

            signs_m = _np.empty((3,), dtype=_np.int32)
            signs_n = _np.empty((3,), dtype=_np.int32)
            for i, (em, en) in enumerate(zip(edges_m, edges_n)):
                signs_m[i] = meshdata.edge_facet_signs[em, meshdata.edge2facet[em] == facet_P][0]
                signs_n[i] = meshdata.edge_facet_signs[en, meshdata.edge2facet[en] == facet_Q][0]

            rows = [edges_m[i // 3] for i in range(9)]
            cols = [edges_n[i % 3] for i in range(9)]

            _demcem_bindings.ws_ea_rwg(
                meshdata.vert_coords[vert_1], meshdata.vert_coords[vert_2],
                meshdata.vert_coords[vert_3], meshdata.vert_coords[vert_4],
                k0, N_quad, N_quad, sing_vals
            )
            L_rows += rows
            L_cols += cols
            L_vals += [
                sing_vals[i] / 4 / _np.pi * signs_m[i // 3] * signs_n[i % 3]
                / meshdata.edge_lengths[edges_m[i // 3]]
                / meshdata.edge_lengths[edges_n[i % 3]]
                for i in range(9)
            ]

            # Symmetric part
            L_rows += cols
            L_cols += rows
            L_vals += L_vals[-9:]

            _demcem_bindings.ss_ea_rwg(
                meshdata.vert_coords[vert_1], meshdata.vert_coords[vert_2],
                meshdata.vert_coords[vert_3], meshdata.vert_coords[vert_4],
                k0, N_quad, N_quad, sing_vals
            )
            K_rows += rows
            K_cols += cols
            K_vals += [
                - sing_vals[i] / 4 / _np.pi * signs_m[i // 3] * signs_n[i % 3]
                / meshdata.edge_lengths[edges_m[i // 3]]
                / meshdata.edge_lengths[edges_n[i % 3]]
                for i in range(9)
            ]

            # Symmetric part
            K_rows += cols
            K_cols += rows
            K_vals += K_vals[-9:]

            for i in range(9):
                singular_entries[(edges_m[i // 3], edges_n[i % 3])] = True
                singular_entries[(edges_n[i // 3], edges_m[i % 3])] = True


    # Compute vertex adjacent terms
    for vert_1 in range(meshdata.vert_coords.shape[0]):
        for facet_P in vert2facet.links(vert_1):
            for facet_Q in vert2facet.links(vert_1):

                # Each facet pair is only done once due to symmetry
                if (max(facet_P, facet_Q), min(facet_P, facet_Q)) in facetpairs_done:
                    continue

                facetpairs_done.add((max(facet_P, facet_Q), min(facet_P, facet_Q)))

                # Non-common vertex indices according to DEMCEM (not adjusted for normal flips)
                loc_common_P = _np.nonzero(meshdata.facet2vert[facet_P] == vert_1)[0][0]
                loc_common_Q = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_1)[0][0]
                vert_2 = meshdata.facet2vert[facet_P][(loc_common_P + 1) % 3]
                vert_3 = meshdata.facet2vert[facet_P][(loc_common_P + 2) % 3]
                vert_5 = meshdata.facet2vert[facet_Q][(loc_common_Q + 1) % 3]
                vert_4 = meshdata.facet2vert[facet_Q][(loc_common_Q + 2) % 3]

                # Switch order of non-common vertices if this does not correspond to ordering
                # associated with positive outward normal.
                if meshdata.facet_flips[facet_P] < 0:
                    vert_2, vert_3 = vert_3, vert_2
                if meshdata.facet_flips[facet_Q] < 0:
                    vert_4, vert_5 = vert_5, vert_4

                local_P = _np.empty(3, dtype=_np.int32)
                local_P[0] = _np.nonzero(meshdata.facet2vert[facet_P] == vert_1)[0][0]
                local_P[1] = _np.nonzero(meshdata.facet2vert[facet_P] == vert_2)[0][0]
                local_P[2] = _np.nonzero(meshdata.facet2vert[facet_P] == vert_3)[0][0]
                local_Q = _np.empty(3, dtype=_np.int32)
                local_Q[0] = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_1)[0][0]
                local_Q[1] = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_4)[0][0]
                local_Q[2] = _np.nonzero(meshdata.facet2vert[facet_Q] == vert_5)[0][0]

                # Edges reordered according to DEMCEM local indexation
                edges_m = meshdata.facet2edge[facet_P][local_P]
                edges_n = meshdata.facet2edge[facet_Q][local_Q]

                signs_m = _np.empty((3,), dtype=_np.int32)
                signs_n = _np.empty((3,), dtype=_np.int32)
                for i, (em, en) in enumerate(zip(edges_m, edges_n)):
                    signs_m[i] = meshdata.edge_facet_signs[em, meshdata.edge2facet[em] == facet_P][0]
                    signs_n[i] = meshdata.edge_facet_signs[en, meshdata.edge2facet[en] == facet_Q][0]

                rows = [edges_m[i // 3] for i in range(9)]
                cols = [edges_n[i % 3] for i in range(9)]

                _demcem_bindings.ws_va_rwg(
                    meshdata.vert_coords[vert_1], meshdata.vert_coords[vert_2],
                    meshdata.vert_coords[vert_3], meshdata.vert_coords[vert_4],
                    meshdata.vert_coords[vert_5], k0, N_quad, N_quad, N_quad, sing_vals
                )
                L_rows += rows
                L_cols += cols
                L_vals += [
                    sing_vals[i] / 4 / _np.pi * signs_m[i // 3] * signs_n[i % 3]
                    / meshdata.edge_lengths[edges_m[i // 3]]
                    / meshdata.edge_lengths[edges_n[i % 3]]
                    for i in range(9)
                ]

                # Symmetric part
                L_rows += cols
                L_cols += rows
                L_vals += L_vals[-9:]

                _demcem_bindings.ss_va_rwg(
                    meshdata.vert_coords[vert_1], meshdata.vert_coords[vert_2],
                    meshdata.vert_coords[vert_3], meshdata.vert_coords[vert_4],
                    meshdata.vert_coords[vert_5], k0, N_quad, N_quad, N_quad, sing_vals
                )
                K_rows += rows
                K_cols += cols
                K_vals += [
                    - sing_vals[i] / 4 / _np.pi * signs_m[i // 3] * signs_n[i % 3]
                    / meshdata.edge_lengths[edges_m[i // 3]]
                    / meshdata.edge_lengths[edges_n[i % 3]]
                    for i in range(9)
                ]

                # Symmetric part
                K_rows += cols
                K_cols += rows
                K_vals += K_vals[-9:]

                for i in range(9):
                    singular_entries[(edges_m[i // 3], edges_n[i % 3])] = True
                    singular_entries[(edges_n[i // 3], edges_m[i % 3])] = True

    # Compute "leftover" elements included in terms containing singularity but which corresponds to
    # facets neither ST, EA or VA. The method is not very optimized, but should at least not be
    # executed for that many entries since it's only for terms with singularities
    for edge_m, edge_n in singular_entries:

        for i_facet_m, facet_m in enumerate(meshdata.edge2facet[edge_m]):
            for i_facet_n, facet_n in enumerate(meshdata.edge2facet[edge_n]):

                # Edge/vertex adjacent facets and self-facets are already computed
                if (max(facet_m, facet_n), min(facet_m, facet_n)) in facetpairs_done:
                    continue

                K_contrib, L_contrib = _assembly_full.facet_contrib_KL_operators(
                    k0, basisdata.basis, basisdata.divs, basisdata.quad_points,
                    basisdata.quad_weights, meshdata.edge2facet, meshdata.facet_areas,
                    edge_m, edge_n, i_facet_m, i_facet_n
                )
                K_rows.append(edge_m)
                K_cols.append(edge_n)
                K_vals.append(K_contrib)

                L_rows.append(edge_m)
                L_cols.append(edge_n)
                L_vals.append(L_contrib)

    # Compute self-facet terms for the K operator using the Numba assembly for the B matrix block
    B_rows, B_cols, B_vals = _assembly_full.assemble_B_integral(
        basisdata.basis, basisdata.quad_weights,
        meshdata.facet2edge, meshdata.edge2facet, meshdata.facet_areas, meshdata.facet_normals
    )

    if gen_preconditioner:
        K_prec = 0.5 * _sparse.coo_array(
            (B_vals, (B_rows, B_cols)), shape=(num_edges, num_edges)
        ).tocsc()
        L_prec = _sparse.coo_array(
            (_np.array(Lp_vals), (_np.array(Lp_rows), _np.array(Lp_cols))), shape=(num_edges, num_edges)
        ).tocsc()
    else:
        K_prec = None
        L_prec = None

    K_rows = _np.concatenate((K_rows, B_rows))
    del B_rows
    K_cols = _np.concatenate((K_cols, B_cols))
    del B_cols
    K_vals = _np.concatenate((K_vals, 0.5 * B_vals))
    del B_vals

    return K_rows, K_cols, K_vals, L_rows, L_cols, L_vals, singular_entries, K_prec, L_prec
