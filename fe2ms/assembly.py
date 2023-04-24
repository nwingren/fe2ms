"""
Functions for assembly of the FE and BI blocks.

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
from scipy import sparse as _sparse
import dolfinx as _dolfinx
import ufl as _ufl
import fe2ms.bi_space as _bi_space
import fe2ms.assembly_nonsingular_full as _assembly_full
import fe2ms.assembly_nonsingular_aca as _assembly_aca
from fe2ms.utility import(
    ComputationVolume as _ComputationVolume,
    FEBISpaces as _FEBISpaces
)
from adaptoctree import tree as _tree, morton as _morton
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
        zero_fun.interpolate(lambda x: (0*x[0], 0*x[0], 0*x[0]))
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
    inv_mur = computation_volume.get_inv_mur()
    if computation_volume.bi_material:
        xi = computation_volume.get_xi()
        zeta = computation_volume.get_zeta()
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
            *_ufl.dx
        )
    K_matrix = _dolfinx.fem.assemble_matrix(form, bcs=boundary_conditions)

    K_matrix.finalize() # pylint: disable=no-member
    K_matrix = _sparse.csr_array((K_matrix.data, K_matrix.indices, K_matrix.indptr)) # pylint: disable=no-member

    rows, cols, B_vals = _assembly_full.assemble_B(
        k0, function_spaces.bi_basisdata.basis,
        function_spaces.bi_basisdata.quad_points, function_spaces.bi_basisdata.quad_weights,
        function_spaces.bi_meshdata.facet2edge, function_spaces.bi_meshdata.edge2facet,
        function_spaces.bi_meshdata.facet_areas, function_spaces.bi_meshdata.facet_normals
    )
    B_matrix = _sparse.coo_array((B_vals, (rows, cols)), shape=2*(function_spaces.bi_size,)).tocsr()

    return K_matrix, B_matrix


def assemble_bi_blocks_full(
    formulation, k0, meshdata: _bi_space.BIMeshData, basisdata: _bi_space.BIBasisData,
    quad_order_singular
):
    """
    Assemble full BI blocks for tested K- and L-operators. This uses edge-wise assembly as opposed
    to the more traditional facet-wise assembly used in MoM.

    Parameters
    ----------
    formulation : {'is-efie', 'vs-efie', 'ej'}
        Formulation used (affects K-operator).
    k0 : float
        Free-space wavenumber of problem.
    meshdata : febicode.utility.BIMeshData
        Mesh data for the BI problem.
    basisdata : febicode.rwg_rt_helpers.BIBasisData
        Basis data for the BI problem.
    Einc_fun : function
        Function describing an incident electric field. Both its input and output should be ndarray
        with shape (n, 3) where n is an arbitrary number of evaluation points. None as input is
        interpreted as no external incident field.
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

    if formulation in ('is-efie', 'vs-efie'):
        Kop_singular, Lop_singular, singular_entries, K_prec, L_prec = _compute_singularities_KL_operators(
            k0, meshdata, basisdata, quad_order_singular, False, True
        )
    elif formulation == 'ej':
        Kop_singular, Lop_singular, singular_entries, K_prec, L_prec = _compute_singularities_KL_operators(
            k0, meshdata, basisdata, quad_order_singular, True, True
        )
    else:
        raise NotImplementedError(f'Formulation \'{formulation}\' not implemented')

    _assembly_full.assemble_KL_operators(
        k0, basisdata.basis, basisdata.divs, basisdata.quad_points, basisdata.quad_weights,
        meshdata.edge2facet, meshdata.facet_areas, singular_entries, Kop_matrix, Lop_matrix
    )

    Kop_matrix += Kop_singular
    Lop_matrix += Lop_singular

    return Kop_matrix, Lop_matrix, K_prec, L_prec


def assemble_bi_aca(
    formulation, k0, meshdata: _bi_space.BIMeshData, basisdata: _bi_space.BIBasisData,
    recompress, target_points, max_points, max_level, tolerance, quad_order_singular
):
    """
    Assemble BI blocks using the adaptive cross approximation (ACA). Uses AdaptOctree to subdivide
    geometry into near and far parts (far parts are treated in a multilevel fashion). Near parts are
    assembled fully into sparse arrays, and far parts are assembled using the ACA into a series of
    outer product forms. A special object is used to handle operations using the ACA data.

    Parameters
    ----------
    formulation : {'is-efie', 'vs-efie', 'ej'}
        Formulation used (affects K operator).
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

    if formulation in ('is-efie', 'vs-efie'):
        Kop_singular, Lop_singular, singular_entries, K_prec, L_prec = _compute_singularities_KL_operators(
            k0, meshdata, basisdata, quad_order_singular, False, True
        )
    elif formulation == 'ej':
        Kop_singular, Lop_singular, singular_entries, K_prec, L_prec = _compute_singularities_KL_operators(
            k0, meshdata, basisdata, quad_order_singular, True, True
        )
    else:
        raise NotImplementedError(f'Formulation \'{formulation}\' not implemented')

    rows, cols, Kop_vals, Lop_vals = _assembly_aca.compute_KL_operators_near_octree(
        k0, basisdata.basis, basisdata.divs, basisdata.quad_points,
        basisdata.quad_weights, meshdata.edge2facet, meshdata.facet_areas,
        singular_entries, complete_tree, u, edge2key_leaves
    )

    Kop_near = _sparse.coo_array((Kop_vals, (rows, cols)), shape=2*(meshdata.edge2vert.shape[0],)).tocsc()
    Kop_near += Kop_singular
    Lop_near = _sparse.coo_array((Lop_vals, (rows, cols)), shape=2*(meshdata.edge2vert.shape[0],)).tocsc()
    Lop_near += Lop_singular

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
    formulation : {'is-efie', 'vs-efie', 'ej'}
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

    if formulation not in ('is-efie', 'vs-efie', 'ej'):
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

    if formulation == 'ej':
        b_inc = -1j * k0 * _np.concatenate((b_M, b_inc))

    return b_inc


def _compute_singularities_KL_operators(
    k0, meshdata: _bi_space.BIMeshData, basisdata: _bi_space.BIBasisData,
    N_quad, on_interior, gen_preconditioner
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
    on_interior : bool
        Whether operators are on the interior of a computational volume. Affects sign of extracted
        K singularity.
    gen_preconditioner : bool
        Whether to generate sparsified operators for preconditioner. Will add two outputs.

    Returns
    -------
    K_singular : scipy.sparse.csc_array
        K entries containing singular terms (strong singularity).
    L_singular : scipy.sparse.csc_array
        L entries containing singular terms (weak singularity).
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
        Kp_rows = []
        Kp_cols = []
        Lp_rows = []
        Lp_cols = []
        Kp_vals = []
        Lp_vals = []

    # Need extra connectivity for VA
    meshdata.mesh.topology.create_connectivity(0, 2)
    vert2facet = meshdata.mesh.topology.connectivity(0, 2)

    # Compute self-terms
    for facet_P, verts_P in enumerate(meshdata.facet2vert):

        edges_m = meshdata.facet2edge[facet_P]

        sing_vals = _np.zeros(9, dtype=_np.complex128)
        r_coords = meshdata.vert_coords[verts_P]
        _demcem_bindings.ws_st_rwg(
            r_coords[0], r_coords[1], r_coords[2], k0, N_quad, sing_vals
        )

        # Find signs of RWG corresponding to edge_m in facet_P
        # Also add term from nxK to the P values (sign depends on in/outside of volume)
        signs_m = _np.empty((3,), dtype=_np.int32)
        normal_P = meshdata.facet_normals[facet_P]
        J_P = 2 * meshdata.facet_areas[facet_P]
        for i, e in enumerate(edges_m):
            f = meshdata.edge2facet[e] == facet_P
            signs_m[i] = meshdata.edge_facet_signs[e, f][0]

            # Only get non self combinations (self term is identically zero)
            for e_n in edges_m[i+1:]:
                f_n = meshdata.edge2facet[e_n] == facet_P
                cross_product = _np.cross(normal_P, basisdata.basis[e_n, f_n])
                if on_interior:
                    cross_product *= -1
                K_rows.append(e)
                K_cols.append(e_n)
                K_vals.append(-0.5 * _np.sum(
                    basisdata.basis[e, f] * cross_product *
                    basisdata.quad_weights.reshape(-1, 1)
                ) * J_P)
                if gen_preconditioner:
                    Kp_rows.append(K_rows[-1])
                    Kp_cols.append(K_cols[-1])
                    Kp_vals.append(K_vals[-1])

                # Add skew symmetric part
                K_rows.append(e_n)
                K_cols.append(e)
                K_vals.append(-K_vals[-1])
                if gen_preconditioner:
                    Kp_rows.append(K_rows[-1])
                    Kp_cols.append(K_cols[-1])
                    Kp_vals.append(K_vals[-1])

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

        # Find all EA facets (remove self facet)
        # ea_facets is ordered the same way as edges_m
        ea_facets = meshdata.edge2facet[edges_m]
        ea_facets = ea_facets[ea_facets != facet_P]
        for e_local, facet_Q in enumerate(ea_facets):

            # Each facet pair is only done once due to symmetry
            if (max(facet_P, facet_Q), min(facet_P, facet_Q)) in facetpairs_done:
                continue

            facetpairs_done.add((max(facet_P, facet_Q), min(facet_P, facet_Q)))

            # Verts in facet P are cyclically permuted according to the edge local index in facet P
            local_roll_P = _np.roll(_np.arange(3), 2-e_local)
            if meshdata.facet_flips[facet_P] == -1:
                local_roll_P[[0, 1]] = local_roll_P[[1, 0]]
            edges_m_adj = edges_m[local_roll_P]

            verts_Q = meshdata.facet2vert[facet_Q]
            edges_n = meshdata.facet2edge[facet_Q]

            # Verts in facet Q are permuted in a more complicated way
            local_roll_Q = _np.empty(3, dtype=_np.int32)
            local_roll_Q[0] = _np.where(verts_Q == verts_P[local_roll_P][1])[0]
            local_roll_Q[1] = _np.where(verts_Q == verts_P[local_roll_P][0])[0]
            local_roll_Q[2] = _np.setdiff1d(_np.arange(3), local_roll_Q[:2])
            edges_n_adj = edges_n[local_roll_Q]

            r_coords_adj = r_coords[local_roll_P]
            r4 = meshdata.vert_coords[verts_Q[local_roll_Q][2]]

            # Find signs of RWG corresponding to edge_n in facet_Q. Also roll signs_m
            signs_m_adj = signs_m[local_roll_P]
            signs_n_adj = _np.empty((3,), dtype=_np.int32)
            for i, e in enumerate(edges_n_adj):
                signs_n_adj[i] = meshdata.edge_facet_signs[e, meshdata.edge2facet[e] == facet_Q][0]
            
            rows = [edges_m_adj[i // 3] for i in range(9)]
            cols = [edges_n_adj[i % 3] for i in range(9)]

            _demcem_bindings.ws_ea_rwg(
                r_coords_adj[0], r_coords_adj[1], r_coords_adj[2], r4, k0,
                N_quad, N_quad, sing_vals
            )
            L_rows += rows
            L_cols += cols
            L_vals += [
                sing_vals[i] / 4 / _np.pi * signs_m_adj[i // 3] * signs_n_adj[i % 3]
                / meshdata.edge_lengths[edges_m_adj[i // 3]]
                / meshdata.edge_lengths[edges_n_adj[i % 3]]
                for i in range(9)
            ]

            # Symmetric part
            L_rows += cols
            L_cols += rows
            L_vals += L_vals[-9:]

            _demcem_bindings.ss_ea_rwg(
                r_coords_adj[0], r_coords_adj[1], r_coords_adj[2], r4, k0,
                N_quad, N_quad, sing_vals
            )
            K_rows += rows
            K_cols += cols
            K_vals += [
                - sing_vals[i] / 4 / _np.pi * signs_m_adj[i // 3] * signs_n_adj[i % 3]
                / meshdata.edge_lengths[edges_m_adj[i // 3]]
                / meshdata.edge_lengths[edges_n_adj[i % 3]]
                for i in range(9)
            ]

            # Symmetric part
            K_rows += cols
            K_cols += rows
            K_vals += K_vals[-9:]
            
            for i in range(9):
                singular_entries[(edges_m_adj[i // 3], edges_n_adj[i % 3])] = True
                singular_entries[(edges_n_adj[i % 3], edges_m_adj[i // 3])] = True

        # Find all VA facets
        for v in verts_P:
            for facet_Q in vert2facet.links(v):

                # Each facet pair is only done once due to symmetry
                if (max(facet_P, facet_Q), min(facet_P, facet_Q)) in facetpairs_done:
                    continue

                facetpairs_done.add((max(facet_P, facet_Q), min(facet_P, facet_Q)))

                # Verts in facet P are permuted according to the local index of common vert
                v_loc_P = _np.where(verts_P == v)[0][0]
                local_roll_P = _np.roll(_np.arange(3), -v_loc_P)
                if meshdata.facet_flips[facet_P] == -1:
                    local_roll_P[[1, 2]] = local_roll_P[[2, 1]]
                edges_m_adj = edges_m[local_roll_P]

                verts_Q = meshdata.facet2vert[facet_Q]
                edges_n = meshdata.facet2edge[facet_Q]

                # Similar for verts in facet Q but there is also a flip
                verts_Q = meshdata.facet2vert[facet_Q]
                v_loc_Q = _np.where(verts_Q == v)[0][0]
                local_roll_Q = _np.roll(_np.arange(3), -v_loc_Q)
                if meshdata.facet_flips[facet_Q] == 1:
                    local_roll_Q[[1, 2]] = local_roll_Q[[2, 1]]
                edges_n_adj = edges_n[local_roll_Q]

                r_coords_adj = r_coords[local_roll_P]
                r4 = meshdata.vert_coords[verts_Q[local_roll_Q][1]]
                r5 = meshdata.vert_coords[verts_Q[local_roll_Q][2]]

                # Find signs of RWG corresponding to edge_n in facet_Q. Also roll signs_m
                signs_m_adj = signs_m[local_roll_P]
                signs_n_adj = _np.empty((3,), dtype=_np.int32)
                for i, e in enumerate(edges_n_adj):
                    signs_n_adj[i] = meshdata.edge_facet_signs[e, meshdata.edge2facet[e] == facet_Q][0]

                rows = [edges_m_adj[i // 3] for i in range(9)]
                cols = [edges_n_adj[i % 3] for i in range(9)]

                _demcem_bindings.ws_va_rwg(
                    r_coords_adj[0], r_coords_adj[1], r_coords_adj[2], r4, r5, k0,
                    N_quad, N_quad, N_quad, sing_vals
                )
                L_rows += rows
                L_cols += cols
                L_vals += [
                    sing_vals[i] / 4 / _np.pi * signs_m_adj[i // 3] * signs_n_adj[i % 3]
                    / meshdata.edge_lengths[edges_m_adj[i // 3]]
                    / meshdata.edge_lengths[edges_n_adj[i % 3]]
                    for i in range(9)
                ]

                # Symmetric part
                L_rows += cols
                L_cols += rows
                L_vals += L_vals[-9:]

                _demcem_bindings.ss_va_rwg(
                    r_coords_adj[0], r_coords_adj[1], r_coords_adj[2], r4, r5, k0,
                    N_quad, N_quad, N_quad, sing_vals
                )
                K_rows += rows
                K_cols += cols
                K_vals += [
                    - sing_vals[i] / 4 / _np.pi * signs_m_adj[i // 3] * signs_n_adj[i % 3]
                    / meshdata.edge_lengths[edges_m_adj[i // 3]]
                    / meshdata.edge_lengths[edges_n_adj[i % 3]]
                    for i in range(9)
                ]

                # Symmetric part
                K_rows += cols
                K_cols += rows
                K_vals += K_vals[-9:]

                for i in range(9):
                    singular_entries[(edges_m_adj[i // 3], edges_n_adj[i % 3])] = True
                    singular_entries[(edges_n_adj[i % 3], edges_m_adj[i // 3])] = True

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
    
    K_singular = _sparse.coo_array(
        (_np.array(K_vals), (_np.array(K_rows), _np.array(K_cols))), shape=(num_edges, num_edges)
    ).tocsc()
    L_singular = _sparse.coo_array(
        (_np.array(L_vals), (_np.array(L_rows), _np.array(L_cols))), shape=(num_edges, num_edges)
    ).tocsc()

    if gen_preconditioner:
        K_prec = _sparse.coo_array(
            (_np.array(Kp_vals), (_np.array(Kp_rows), _np.array(Kp_cols))), shape=(num_edges, num_edges)
        ).tocsc()
        L_prec = _sparse.coo_array(
            (_np.array(Lp_vals), (_np.array(Lp_rows), _np.array(Lp_cols))), shape=(num_edges, num_edges)
        ).tocsc()
    else:
        K_prec = None
        L_prec = None

    return K_singular, L_singular, singular_entries, K_prec, L_prec
