"""
Internal assembly functions for ACA nonsingular matrix assembly with numba JIT compilation.

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
from adaptoctree import morton as _morton

@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def compute_KL_operators_near_octree(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
    singular_entries, complete_tree, u_array, edge2key_leaves
):
    """
    Compute near, nonsingular entries for tested K- and L-operators with parallelization.
    Nearness determined using an octree where a leaf group (source) will be considered to be near
    all neighbor groups (observers), including itself.
    Neighbor groups are listed in u_array for all octree groups.
    Return values are arrays rows, cols, Kop, Lop values for coo_array construction.
    """

    num_entries = 0
    rows = []
    cols = []
    Kop_vals = []
    Lop_vals = []

    for source_index in range(u_array.shape[0]):

        # Assemble self-group interaction
        if complete_tree[source_index] in edge2key_leaves:
            edges_n = _np.where(edge2key_leaves == complete_tree[source_index])[0]
            rows_block = _np.empty(edges_n.shape[0]**2, dtype=_np.int64)
            cols_block = _np.empty(edges_n.shape[0]**2, dtype=_np.int64)
            for i in range(edges_n.shape[0]):
                for j in range(edges_n.shape[0]):
                    rows_block[i*edges_n.shape[0] + j] = edges_n[i]
                    cols_block[i*edges_n.shape[0] + j] = edges_n[j]
            Kop_block = _np.zeros(rows_block.shape, dtype=_np.complex128)
            Lop_block = _np.zeros(rows_block.shape, dtype=_np.complex128)
            _compute_KL_operator_entries_near(
                k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                singular_entries, Kop_block, Lop_block, rows_block, cols_block
            )
            rows.append(rows_block)
            cols.append(cols_block)
            Kop_vals.append(Kop_block)
            Lop_vals.append(Lop_block)
            num_entries += rows_block.shape[0]

            for obs_index in range(u_array.shape[1]):
                if u_array[source_index, obs_index] != -1:
                    edges_m = _np.where(
                        edge2key_leaves == u_array[source_index, obs_index]
                    )[0]
                    rows_block = _np.empty(edges_n.shape[0]*edges_m.shape[0], dtype=_np.int64)
                    cols_block = _np.empty(edges_n.shape[0]*edges_m.shape[0], dtype=_np.int64)
                    for i in range(edges_m.shape[0]):
                        for j in range(edges_n.shape[0]):
                            rows_block[i*edges_n.shape[0] + j] = edges_m[i]
                            cols_block[i*edges_n.shape[0] + j] = edges_n[j]
                    Kop_block = _np.zeros(rows_block.shape, dtype=_np.complex128)
                    Lop_block = _np.zeros(rows_block.shape, dtype=_np.complex128)
                    _compute_KL_operator_entries_near(
                        k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                        singular_entries, Kop_block, Lop_block, rows_block, cols_block
                    )
                    rows.append(rows_block)
                    cols.append(cols_block)
                    Kop_vals.append(Kop_block)
                    Lop_vals.append(Lop_block)
                    num_entries += rows_block.shape[0]

    rows_array = _np.zeros(num_entries, dtype=_np.int64)
    cols_array = _np.zeros(num_entries, dtype=_np.int64)
    Kop_array = _np.zeros(num_entries, dtype=_np.complex128)
    Lop_array = _np.zeros(num_entries, dtype=_np.complex128)
    ind = 0

    for i in range(len(rows)): # pylint: disable=consider-using-enumerate
        rows_array[ind:ind+len(rows[i])] = rows[i]
        cols_array[ind:ind+len(rows[i])] = cols[i]
        Kop_array[ind:ind+len(rows[i])] = Kop_vals[i]
        Lop_array[ind:ind+len(rows[i])] = Lop_vals[i]
        ind += len(rows[i])

    return rows_array, cols_array, Kop_array, Lop_array


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def find_near_indices_octree(complete_tree, u_array, edge2key_leaves, singular_entries):
    """
    Find indices of near matrix entries based on octree near neighbors.
    Will not compute any entries, so it is useful only for cases where entries are available, like
    sparsification of near Kop and Lop for preconditioning.
    singular_entries is needed to ensure that they are included in the resulting set of indices.
    """

    near_set = set()

    for source_index in range(u_array.shape[0]):

        # Find self-group interaction
        if complete_tree[source_index] in edge2key_leaves:
            edges_n = _np.where(edge2key_leaves == complete_tree[source_index])[0]
            for i in range(edges_n.shape[0]):
                for j in range(edges_n.shape[0]):
                    near_set.add((_nb.int32(edges_n[i]), _nb.int32(edges_n[j])))

            for obs_index in range(u_array.shape[1]):
                if u_array[source_index, obs_index] != -1:
                    edges_m = _np.where(
                        edge2key_leaves == u_array[source_index, obs_index]
                    )[0]
                    for i in range(edges_m.shape[0]):
                        for j in range(edges_n.shape[0]):
                            near_set.add((_nb.int32(edges_m[i]), _nb.int32(edges_n[j])))

    near_set.update(set(singular_entries.keys()))
    rows_array = _np.empty(len(near_set), dtype=_np.int32)
    cols_array = _np.empty(len(near_set), dtype=_np.int32)

    for i, e in enumerate(near_set): # pylint: disable=consider-using-enumerate
        rows_array[i] = e[0]
        cols_array[i] = e[1]

    return rows_array, cols_array


@_nb.experimental.jitclass([
    ('edge_list_m', _nb.types.List(_nb.types.Array(dtype=_nb.int64, ndim=1, layout='C'))),
    ('edge_list_n', _nb.types.List(_nb.types.Array(dtype=_nb.int64, ndim=1, layout='C'))),
    ('U_list_Kop', _nb.types.List(_nb.types.Array(dtype=_nb.complex128, ndim=2, layout='C'))),
    ('U_list_Lop', _nb.types.List(_nb.types.Array(dtype=_nb.complex128, ndim=2, layout='C'))),
    ('Vh_list_Kop', _nb.types.List(_nb.types.Array(dtype=_nb.complex128, ndim=2, layout='C'))),
    ('Vh_list_Lop', _nb.types.List(_nb.types.Array(dtype=_nb.complex128, ndim=2, layout='C')))
])
class MultilevelOperator:
    """
    Class for containing multilevel Kop and Lop representation and performing matrix-vector
    multiplication using this.
    """

    def __init__(self, edge_list_m, edge_list_n, U_list_Kop, U_list_Lop, Vh_list_Kop, Vh_list_Lop):
        self.edge_list_m = edge_list_m
        self.edge_list_n = edge_list_n
        self.U_list_Kop = U_list_Kop
        self.U_list_Lop = U_list_Lop
        self.Vh_list_Kop = Vh_list_Kop
        self.Vh_list_Lop = Vh_list_Lop
    
    def matvec_Kop(self, x):
        """
        Compute matrix-vector product for Kop.
        """
        y = _np.zeros_like(x)
        for i in range(len(self.edge_list_m)): # pylint: disable=consider-using-enumerate
            y[self.edge_list_m[i]] += \
                self.U_list_Kop[i] @ (self.Vh_list_Kop[i] @ x[self.edge_list_n[i]])
        return y

    def matvec_Kop_T(self, x):
        """
        Compute matrix-vector product for transpose of Kop.
        """
        y = _np.zeros_like(x)
        for i in range(len(self.edge_list_m)): # pylint: disable=consider-using-enumerate
            y[self.edge_list_n[i]] += \
                self.Vh_list_Kop[i].T @ (self.U_list_Kop[i].T @ x[self.edge_list_m[i]])
        return y

    def matvec_Lop(self, x):
        """
        Compute matrix-vector product for Lop.
        """
        y = _np.zeros_like(x)
        for i in range(len(self.edge_list_m)): # pylint: disable=consider-using-enumerate
            y[self.edge_list_m[i]] += \
                self.U_list_Lop[i] @ (self.Vh_list_Lop[i] @ x[self.edge_list_n[i]])
        return y
    
    def matvec_both(self, x_Kop, x_Lop):
        """
        Compute matrix-vector product for both Kop and Lop simultaneously.
        """
        y_Kop = _np.zeros_like(x_Kop)
        y_Lop = _np.zeros_like(x_Lop)
        for i in range(len(self.edge_list_m)): # pylint: disable=consider-using-enumerate
            y_Kop[self.edge_list_m[i]] += \
                self.U_list_Kop[i] @ (self.Vh_list_Kop[i] @ x_Kop[self.edge_list_n[i]])
            y_Lop[self.edge_list_m[i]] += \
                self.U_list_Lop[i] @ (self.Vh_list_Lop[i] @ x_Lop[self.edge_list_n[i]])
        return y_Kop, y_Lop
    
    def approx_size(self):
        """
        Compute approximate size in bytes.
        """
        mem = 0.
        for i in range(len(self.edge_list_m)): # pylint: disable=consider-using-enumerate
            mem += 8 * (self.edge_list_m[i].size + self.edge_list_n[i].size)
            mem += 16 * (self.U_list_Kop[i].size + self.Vh_list_Kop[i].size)
            mem += 16 * (self.U_list_Lop[i].size + self.Vh_list_Lop[i].size)
        
        return mem


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def compute_KL_operators_far_octree(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas, complete_tree,
    x_array, v_array, w_array, edge2key_levels, epsilon, recompress
):
    # TODO: Remove edge2key_levels and compute those things in a different way?

    edge_list_m = []
    edge_list_m.append(_np.empty(1, dtype=_np.int64))
    edge_list_m.clear()

    edge_list_n = edge_list_m.copy()

    U_list_Kop = []
    U_list_Kop.append(_np.empty((1,1), dtype=_np.complex128))
    U_list_Kop.clear()

    Vh_list_Kop = U_list_Kop.copy()
    U_list_Lop = U_list_Kop.copy()
    Vh_list_Lop = U_list_Kop.copy()

    # TODO: See if it's possible to use prange in one of these loops
    # (it's not worth it at lower levels)
    for source_index in range(complete_tree.shape[0]):

        source_level = _morton.find_level(complete_tree[source_index])

        if complete_tree[source_index] not in edge2key_levels[source_level]:
            continue

        # Interaction lists in adaptree seem to break for level 0 (this level means everything)
        if source_level == 0:
            continue
        edges_n = _np.where(edge2key_levels[source_level] == complete_tree[source_index])[0]

        for obs_index in range(x_array.shape[1]):
            if x_array[source_index, obs_index] != -1:
                obs_level = _morton.find_level(x_array[source_index, obs_index])
                if x_array[source_index, obs_index] not in edge2key_levels[obs_level]:
                    continue
                edges_m = _np.where(
                    edge2key_levels[obs_level] == x_array[source_index, obs_index]
                )[0]
                edge_list_n.append(edges_n)
                edge_list_m.append(edges_m)

                U, Vh = _aca_efie(
                    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                    edges_m, edges_n, compute_K_operator_entry_far, epsilon
                )
                if recompress:
                    U, Vh = recompress_uv(U, Vh, epsilon)
                U_list_Kop.append(U)
                Vh_list_Kop.append(Vh)

                U, Vh = _aca_efie(
                    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                    edges_m, edges_n, compute_L_operator_entry_far, epsilon
                )
                if recompress:
                    U, Vh = recompress_uv(U, Vh, epsilon)
                U_list_Lop.append(U)
                Vh_list_Lop.append(Vh)

        for obs_index in range(v_array.shape[1]):
            if v_array[source_index, obs_index] != -1:
                obs_level = _morton.find_level(v_array[source_index, obs_index])
                if v_array[source_index, obs_index] not in edge2key_levels[obs_level]:
                    continue
                edges_m = _np.where(
                    edge2key_levels[obs_level] == v_array[source_index, obs_index]
                )[0]
                edge_list_n.append(edges_n)
                edge_list_m.append(edges_m)

                U, Vh = _aca_efie(
                    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                    edges_m, edges_n, compute_K_operator_entry_far, epsilon
                )
                if recompress:
                    U, Vh = recompress_uv(U, Vh, epsilon)
                U_list_Kop.append(U)
                Vh_list_Kop.append(Vh)

                U, Vh = _aca_efie(
                    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                    edges_m, edges_n, compute_L_operator_entry_far, epsilon
                )
                if recompress:
                    U, Vh = recompress_uv(U, Vh, epsilon)
                U_list_Lop.append(U)
                Vh_list_Lop.append(Vh)

        for obs_index in range(w_array.shape[1]):
            if w_array[source_index, obs_index] != -1:
                obs_level = _morton.find_level(w_array[source_index, obs_index])
                if w_array[source_index, obs_index] not in edge2key_levels[obs_level]:
                    continue
                edges_m = _np.where(
                    edge2key_levels[obs_level] == w_array[source_index, obs_index]
                )[0]
                edge_list_n.append(edges_n)
                edge_list_m.append(edges_m)

                U, Vh = _aca_efie(
                    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                    edges_m, edges_n, compute_K_operator_entry_far, epsilon
                )
                if recompress:
                    U, Vh = recompress_uv(U, Vh, epsilon)
                U_list_Kop.append(U)
                Vh_list_Kop.append(Vh)

                U, Vh = _aca_efie(
                    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                    edges_m, edges_n, compute_L_operator_entry_far, epsilon
                )
                if recompress:
                    U, Vh = recompress_uv(U, Vh, epsilon)
                U_list_Lop.append(U)
                Vh_list_Lop.append(Vh)

    return MultilevelOperator(
        edge_list_m, edge_list_n, U_list_Kop, U_list_Lop, Vh_list_Kop, Vh_list_Lop
    )


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def compute_KL_operators_near_thres(
    k0, basis, divs, quad_points, quad_weights, edge2facet, edge2vert, facet_areas, vert_coords,
    dist_threshold, near_entries
):
    """
    Compute near, nonsingular matrix entries for tested K- and L-operators with parallellization.
    Nearness determined using a distance threshold for edge-to-edge center distance. A typed dict
    near_entries updates (originally containing singular term indices) with indices for computed
    entries. Return values are arrays rows, cols, Kop and Lop values for coo_array construction.
    """

    rows = [_np.int32(x) for x in range(0)]
    cols = [_np.int32(x) for x in range(0)]
    dist_threshold2 = dist_threshold**2

    # Near entries must be computed serially here (as opposed to in the parallel loop later)
    # since appending to lists is not a thread safe operation
    for edge_m in range(_nb.int32(edge2vert.shape[0])):
        for edge_n in range(_nb.int32(edge2vert.shape[0])):

            if (edge_m, edge_n) in near_entries:
                continue

            dist2 = 0.
            for i_coord in range(3):
                dist2 += 0.25 * (
                    vert_coords[edge2vert[edge_m, 1], i_coord]
                    + vert_coords[edge2vert[edge_m, 0], i_coord]
                    - vert_coords[edge2vert[edge_n, 1], i_coord]
                    - vert_coords[edge2vert[edge_n, 0], i_coord]
                )**2

            if dist2 < dist_threshold2:
                rows.append(edge_m)
                cols.append(edge_n)
                near_entries[(edge_m, edge_n)] = True

    Kop_vals = _np.zeros(len(rows), dtype=_np.complex128)
    Lop_vals = _np.zeros(len(rows), dtype=_np.complex128)
    rows = _np.array(rows)
    cols = _np.array(cols)

    no_entry = near_entries.copy()
    no_entry.clear()
    _compute_KL_operator_entries_near(
        k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
        no_entry, Kop_vals, Lop_vals, rows, cols
    )

    return rows, cols, Kop_vals, Lop_vals


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def compute_K_operator_entry_far(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas, edge_m, edge_n, Kop_entry
):
    """
    Assemble far nonsingular entry [edge_m, edge_n] for tested K-operator only.
    This is done in-place, so Kop_entry needs to be slice of shape (1,) for changes to
    be applied to original array.
    """

    for i_facet_m in range(2):
        for i_facet_n in range(2):
            jacobians = (
                4 * facet_areas[edge2facet[edge_n, i_facet_n]]
                * facet_areas[edge2facet[edge_m, i_facet_m]]
            )

            for quad_point_m in range(quad_points.shape[1]):
                for quad_point_n in range(quad_points.shape[1]):

                    dist_squared = 0.
                    for i_coord in range(3):
                        p1 = quad_points[
                            edge2facet[edge_m, i_facet_m], quad_point_m, i_coord
                        ]
                        p2 = quad_points[
                            edge2facet[edge_n, i_facet_n], quad_point_n, i_coord
                        ]
                        dist_squared += p1**2 + p2**2 - 2 * p1 * p2
                    dist = _np.sqrt(dist_squared)

                    cross_factor = -(1j * k0 + 1 / dist) * \
                        _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist_squared

                    # Loop over coordinates for cross product
                    for i_coord in range(3):

                        cross_comp = (
                            quad_points[
                                edge2facet[edge_m, i_facet_m], quad_point_m, (i_coord+1)%3
                            ]
                            - quad_points[
                                edge2facet[edge_n, i_facet_n], quad_point_n, (i_coord+1)%3
                            ]
                        )
                        cross_2 = (
                            quad_points[
                                edge2facet[edge_m, i_facet_m], quad_point_m, (i_coord-1)%3
                            ]
                            - quad_points[
                                edge2facet[edge_n, i_facet_n], quad_point_n, (i_coord-1)%3
                            ]
                        )
                        cross_comp = (
                            cross_comp * basis[edge_n, i_facet_n, quad_point_n, (i_coord-1)%3]
                            - cross_2 * basis[edge_n, i_facet_n, quad_point_n, (i_coord+1)%3]
                        )

                        Kop_entry[0] -= (
                            basis[edge_m, i_facet_m, quad_point_m, i_coord]
                            * cross_factor * cross_comp
                            * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                            * jacobians
                        )


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def compute_L_operator_entry_far(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas, edge_m, edge_n, Lop_entry
):
    """
    Assemble far nonsingular entry [edge_m, edge_n] for tested L-operator only.
    This is done in-place, so Lop_entry needs to be slice of shape (1,) for changes to
    be applied to original array.
    """

    for i_facet_m in range(2):
        for i_facet_n in range(2):
            jacobians = (
                4 * facet_areas[edge2facet[edge_n, i_facet_n]]
                * facet_areas[edge2facet[edge_m, i_facet_m]]
            )

            for quad_point_m in range(quad_points.shape[1]):
                for quad_point_n in range(quad_points.shape[1]):

                    dist_squared = 0.
                    for i_coord in range(3):
                        p1 = quad_points[
                            edge2facet[edge_m, i_facet_m], quad_point_m, i_coord
                        ]
                        p2 = quad_points[
                            edge2facet[edge_n, i_facet_n], quad_point_n, i_coord
                        ]
                        dist_squared += p1**2 + p2**2 - 2 * p1 * p2
                    dist = _np.sqrt(dist_squared)

                    scalar_product = 0j

                    # Loop over coordinates for scalar product
                    for i_coord in range(3):
                        scalar_product += (
                            basis[edge_m, i_facet_m, quad_point_m, i_coord]
                            * basis[edge_n, i_facet_n, quad_point_n, i_coord]
                        )

                    Lop_entry[0] += (
                        1j * (
                            k0 * scalar_product
                            - 1 / k0 * divs[edge_m, i_facet_m] * divs[edge_n, i_facet_n]
                        )
                        * _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist
                        * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                        * jacobians
                    )


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def recompress_uv(U, Vh, tolerance):

    qu, ru = _np.linalg.qr(U)
    qv, rv = _np.linalg.qr(Vh.conj().T)
    u1, s, v1h = _np.linalg.svd(ru @ rv.conj().T)

    # Reduce dimensions according to truncation
    keep_ind = s / s.max() > tolerance
    s = s[keep_ind]
    v1h = v1h[keep_ind, :]
    u1 = u1[:, keep_ind]

    un = qu @ u1
    vn = qv @ (s.reshape(-1,1) * v1h).conj().T
    return _np.ascontiguousarray(un), _np.ascontiguousarray(vn.conj().T)


@_nb.jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def _compute_KL_operator_entries_near(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
    singular_entries, Kop_vals, Lop_vals, edges_m, edges_n
):

    for i_val in _nb.prange(Kop_vals.shape[0]): # pylint: disable=not-an-iterable

        if (_nb.int32(edges_m[i_val]), _nb.int32(edges_n[i_val])) in singular_entries:
            continue

        for i_facet_m in range(2):
            for i_facet_n in range(2):
                jacobians = (
                    4 * facet_areas[edge2facet[edges_n[i_val], i_facet_n]]
                    * facet_areas[edge2facet[edges_m[i_val], i_facet_m]]
                )

                for quad_point_m in range(quad_points.shape[1]):
                    for quad_point_n in range(quad_points.shape[1]):

                        dist_squared = 0.
                        for i_coord in range(3):
                            p1 = quad_points[
                                edge2facet[edges_m[i_val], i_facet_m], quad_point_m, i_coord
                            ]
                            p2 = quad_points[
                                edge2facet[edges_n[i_val], i_facet_n], quad_point_n, i_coord
                            ]
                            dist_squared += p1**2 + p2**2 - 2 * p1 * p2
                        dist = _np.sqrt(dist_squared)


                        scalar_product = 0j
                        cross_factor = -(1j * k0 + 1 / dist) * \
                            _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist_squared

                        # Loop over coordinates for scalar/cross products
                        for i_coord in range(3):
                            scalar_product += (
                                basis[edges_m[i_val], i_facet_m, quad_point_m, i_coord]
                                * basis[edges_n[i_val], i_facet_n, quad_point_n, i_coord]
                            )

                            cross_comp = (
                                quad_points[
                                    edge2facet[edges_m[i_val], i_facet_m], quad_point_m, (i_coord+1)%3
                                ]
                                - quad_points[
                                    edge2facet[edges_n[i_val], i_facet_n], quad_point_n, (i_coord+1)%3
                                ]
                            )
                            cross_2 = (
                                quad_points[
                                    edge2facet[edges_m[i_val], i_facet_m], quad_point_m, (i_coord-1)%3
                                ]
                                - quad_points[
                                    edge2facet[edges_n[i_val], i_facet_n], quad_point_n, (i_coord-1)%3
                                ]
                            )
                            cross_comp = (
                                cross_comp * basis[edges_n[i_val], i_facet_n, quad_point_n, (i_coord-1)%3]
                                - cross_2 * basis[edges_n[i_val], i_facet_n, quad_point_n, (i_coord+1)%3]
                            )

                            Kop_vals[i_val] -= (
                                basis[edges_m[i_val], i_facet_m, quad_point_m, i_coord]
                                * cross_factor * cross_comp
                                * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                                * jacobians
                            )

                        Lop_vals[i_val] += (
                            1j * (
                                k0 * scalar_product
                                - 1 / k0 * divs[edges_m[i_val], i_facet_m] * divs[edges_n[i_val], i_facet_n]
                            )
                            * _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist
                            * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                            * jacobians
                        )


# Here fastmath=False because otherwise the ACA will result in significantly different U, Vh
# (although the difference in U@Vh is not very large). Reason could be poor vectorization of loops
@_nb.jit(nopython=True, error_model='numpy')
def _aca_efie(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
    edges_m, edges_n, assembly_function, epsilon
):
    # Set row index 0 to 0
    row_indices = set([0])
    Ik = 0
    R_row = _np.zeros(edges_n.shape[0], dtype=_np.complex128)
    R_col =_np.zeros(edges_m.shape[0], dtype=_np.complex128)
    norm_Z_squared = 0.

    # Initialize row 0 of V, find column index 0
    maxval = 0
    col_indices = set()
    Jk = 0
    for col in range(edges_n.shape[0]):
        assembly_function(
            k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
            edges_m[Ik], edges_n[col], R_row[col:col+1]
        )
        if R_row[col].real**2 + R_row[col].imag**2 > maxval:
            maxval = R_row[col].real**2 + R_row[col].imag**2
            Jk = col
    col_indices.add(Jk)
    if(R_row[Jk] == 0):
        V_matrix = [R_row]
    else:
        V_matrix = [R_row / R_row[Jk]]

    # Initialize column 0 of U, find row index 1
    maxval = 0
    for row in range(edges_m.shape[0]):
        assembly_function(
            k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
            edges_m[row], edges_n[Jk], R_col[row:row+1]
        )
        if row not in row_indices and R_col[row].real**2 + R_col[row].imag**2 > maxval:
            maxval = R_col[row].real**2 + R_col[row].imag**2
            Ik = row
    row_indices.add(Ik)
    U_matrix = [R_col]

    # If both the first row and column are all zero, the entire block is zero
    # (for K-matrix which is where this can happen)
    if(R_row[Jk] == 0 and R_col[Ik] == 0):
        return R_col.reshape((-1, 1)), R_row.reshape((1, -1))

    norm_Z_squared += _np.linalg.norm(U_matrix[0])**2 * _np.linalg.norm(V_matrix[0])**2

    # Max number of iterations is the smallest matrix dimension (due to row rank == col rank)
    for k in range(1, min(edges_m.shape[0], edges_n.shape[0])):

        # Compute sum part of row k in R
        R_row = _np.zeros(edges_n.shape[0], dtype=_np.complex128)
        for i in range(k):
            R_row -= U_matrix[i][Ik] * V_matrix[i]

        # Update row k of V, find column index k
        maxval = 0
        for col in range(edges_n.shape[0]):
            assembly_function(
                k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                edges_m[Ik], edges_n[col], R_row[col:col+1]
            )
            if col not in col_indices and R_row[col].real**2 + R_row[col].imag**2 > maxval:
                maxval = R_row[col].real**2 + R_row[col].imag**2
                Jk = col
        col_indices.add(Jk)
        V_matrix.append(R_row / R_row[Jk])

        # Compute sum part of column k in R
        R_col = _np.zeros(edges_m.shape[0], dtype=_np.complex128)
        for i in range(k):
            R_col -= V_matrix[i][Jk] * U_matrix[i]

        # Update column K of U, find row index k+1
        maxval = 0
        for row in range(edges_m.shape[0]):
            assembly_function(
                k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
                edges_m[row], edges_n[Jk], R_col[row:row+1]
            )
            if row not in row_indices and R_col[row].real**2 + R_col[row].imag**2 > maxval:
                maxval = R_col[row].real**2 + R_col[row].imag**2
                Ik = row
        row_indices.add(Ik)
        U_matrix.append(R_col)

        norm_Z_squared += _np.linalg.norm(U_matrix[-1])**2 * _np.linalg.norm(V_matrix[-1])**2
        for i in range(k):
            norm_Z_squared += \
                2 * _np.abs(U_matrix[i] @ U_matrix[-1]) * _np.abs(V_matrix[i] @ V_matrix[-1])

        # Check convergence
        if (
            _np.linalg.norm(U_matrix[-1])**2 * _np.linalg.norm(V_matrix[-1])**2
            <= epsilon**2 * norm_Z_squared
        ):
            break

    U_array = _np.zeros((edges_m.shape[0], len(U_matrix)), dtype=_np.complex128)
    Vh_array = _np.zeros((len(V_matrix), edges_n.shape[0]), dtype=_np.complex128)
    for i in range(len(U_matrix)):
        U_array[:, i] = U_matrix[i]
        Vh_array[i, :] = V_matrix[i]
    return U_array, Vh_array
