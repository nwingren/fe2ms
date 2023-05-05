"""
Internal assembly functions for full nonsingular matrix assembly with numba JIT compilation.

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

@_nb.jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def assemble_KL_operators(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
    singular_entries, Kop_matrix, Lop_matrix
):
    """
    Assemble all nonsingular matrix entries for tested K- and L-operators with parallelization.
    Matrices are modified in-place.
    """

    for edge_m in _nb.prange(Kop_matrix.shape[0]): # pylint: disable=not-an-iterable

        # Symmetric assembly
        for edge_n in range(edge_m, Kop_matrix.shape[0]):

            if (_nb.int32(edge_m), _nb.int32(edge_n)) in singular_entries:
                continue

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
                            cross_comp = 0j
                            cross_factor = -(1j * k0 + 1 / dist) * \
                                _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist_squared

                            # Loop over coordinates for scalar/cross products
                            for i_coord in range(3):
                                scalar_product += (
                                    basis[edge_m, i_facet_m, quad_point_m, i_coord]
                                    * basis[edge_n, i_facet_n, quad_point_n, i_coord]
                                )

                                #  Component i_coord of (r - r') x Lambda_n
                                cross_comp = (
                                    (
                                        quad_points[
                                            edge2facet[edge_m, i_facet_m],
                                            quad_point_m, (i_coord+1)%3
                                        ]
                                        - quad_points[
                                            edge2facet[edge_n, i_facet_n],
                                            quad_point_n, (i_coord+1)%3
                                        ]
                                    ) * basis[edge_n, i_facet_n, quad_point_n, (i_coord-1)%3]
                                    - (
                                        quad_points[
                                            edge2facet[edge_m, i_facet_m],
                                            quad_point_m, (i_coord-1)%3
                                        ]
                                        - quad_points[
                                            edge2facet[edge_n, i_facet_n],
                                            quad_point_n, (i_coord-1)%3
                                        ]
                                    ) * basis[edge_n, i_facet_n, quad_point_n, (i_coord+1)%3]
                                )

                                cross_comp *= (
                                    basis[edge_m, i_facet_m, quad_point_m, i_coord] * cross_factor
                                    * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                                    * jacobians
                                )
                                Kop_matrix[edge_m, edge_n] -= cross_comp
                                Kop_matrix[edge_n, edge_m] -= cross_comp

                            Lop_matrix[edge_m, edge_n] += (
                                1j * (
                                    k0 * scalar_product
                                    - 1 / k0 * divs[edge_m, i_facet_m] * divs[edge_n, i_facet_n]
                                )
                                * _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist
                                * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                                * jacobians
                            )
                            Lop_matrix[edge_n, edge_m] += (
                                1j * (
                                    k0 * scalar_product
                                    - 1 / k0 * divs[edge_m, i_facet_m] * divs[edge_n, i_facet_n]
                                )
                                * _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist
                                * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                                * jacobians
                            )


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def facet_contrib_KL_operators(
    k0, basis, divs, quad_points, quad_weights, edge2facet, facet_areas,
    edge_m, edge_n, i_facet_m, i_facet_n
):
    """
    Compute contributions to tested K- and L-operators for given edge pair and facet pair.
    Use only if necessary, other assemblers are more efficient.
    """

    jacobians = (
        4 * facet_areas[edge2facet[edge_n, i_facet_n]]
        * facet_areas[edge2facet[edge_m, i_facet_m]]
    )
    Kop_contrib = 0j
    Lop_contrib = 0j

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
            cross_comp = 0j
            cross_factor = -(1j * k0 + 1 / dist) * \
                _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist_squared

            # Loop over coordinates for scalar/cross products
            for i_coord in range(3):
                scalar_product += (
                    basis[edge_m, i_facet_m, quad_point_m, i_coord]
                    * basis[edge_n, i_facet_n, quad_point_n, i_coord]
                )

                #  Component i_coord of (r - r') x Lambda_n
                cross_comp = (
                    (
                        quad_points[
                            edge2facet[edge_m, i_facet_m],
                            quad_point_m, (i_coord+1)%3
                        ]
                        - quad_points[
                            edge2facet[edge_n, i_facet_n],
                            quad_point_n, (i_coord+1)%3
                        ]
                    ) * basis[edge_n, i_facet_n, quad_point_n, (i_coord-1)%3]
                    - (
                        quad_points[
                            edge2facet[edge_m, i_facet_m],
                            quad_point_m, (i_coord-1)%3
                        ]
                        - quad_points[
                            edge2facet[edge_n, i_facet_n],
                            quad_point_n, (i_coord-1)%3
                        ]
                    ) * basis[edge_n, i_facet_n, quad_point_n, (i_coord+1)%3]
                )

                Kop_contrib -= (
                    basis[edge_m, i_facet_m, quad_point_m, i_coord]
                    * cross_factor * cross_comp
                    * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                    * jacobians
                )

            Lop_contrib += (
                1j * (
                    k0 * scalar_product
                    - 1 / k0 * divs[edge_m, i_facet_m] * divs[edge_n, i_facet_n]
                )
                * _np.exp(-1j * k0 * dist) / 4 / _np.pi / dist
                * quad_weights[quad_point_m] * quad_weights[quad_point_n]
                * jacobians
            )

    return Kop_contrib, Lop_contrib


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def assemble_B_integral(
    basis, quad_weights, facet2edge, edge2facet, facet_areas, facet_normals
):
    rows = []
    cols = []
    vals = []

    for facet in range(facet2edge.shape[0]):
        jacobian = 2 * facet_areas[facet]
        for i_edge_m in range(3):
            i_facet_m = _np.where(edge2facet[facet2edge[facet, i_edge_m]]==facet)[0][0]
            for i_edge_n in range(i_edge_m, 3):
                integral = 0.
                i_facet_n = _np.where(edge2facet[facet2edge[facet, i_edge_n]]==facet)[0][0]
                for i_quad in range(quad_weights.shape[0]):
                    for i_coord in range(3):
                        integral += facet_normals[facet, i_coord] * (
                            basis[facet2edge[facet, i_edge_m], i_facet_m, i_quad, (i_coord+1)%3]
                            * basis[facet2edge[facet, i_edge_n], i_facet_n, i_quad, (i_coord-1)%3]
                            - basis[facet2edge[facet, i_edge_m], i_facet_m, i_quad, (i_coord-1)%3]
                            * basis[facet2edge[facet, i_edge_n], i_facet_n, i_quad, (i_coord+1)%3]
                        ) * quad_weights[i_quad]

                rows.append(facet2edge[facet, i_edge_m])
                cols.append(facet2edge[facet, i_edge_n])
                vals.append(integral * jacobian)

                # Skew-symmetric part
                rows.append(facet2edge[facet, i_edge_n])
                cols.append(facet2edge[facet, i_edge_m])
                vals.append(-integral * jacobian)

    rows_array = _np.zeros(len(vals), dtype=_np.int64)
    cols_array = _np.zeros(len(vals), dtype=_np.int64)
    B_array = _np.zeros(len(vals))
    for i in range(len(rows)): # pylint: disable=consider-using-enumerate
        rows_array[i] = rows[i]
        cols_array[i] = cols[i]
        B_array[i] = vals[i]

    return rows_array, cols_array, B_array
