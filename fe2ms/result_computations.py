"""
Internal functions for computing results with numba JIT compilation.

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
def compute_near_field_bi(
    k0, edge2facet, facet_areas, quad_points, quad_weights, basis,
    points_near, sol_E_bound, sol_H, E_sc
):
    """
    Compute scattered near field for points in BI region. Scattered field E_sc is modified in-place.
    Uses Volakis, Integral Equation Methods for Electromagnetics (2.63a).
    """

    for i_near in _nb.prange(points_near.shape[0]): # pylint: disable=not-an-iterable
        for edge_m in range(edge2facet.shape[0]):

            for i_facet_m in range(edge2facet.shape[1]):

                # Jacobian norm for integral m
                J_m = 2 * facet_areas[edge2facet[edge_m, i_facet_m]]

                quad_points_phys = quad_points[edge2facet[edge_m, i_facet_m]]

                # No multiplication by eta0 since the solH coeffs are already scaled by that
                for i_quad in range(quad_points_phys.shape[0]):
                    basis_eval = basis[edge_m, i_facet_m, i_quad]
                    Rhat = points_near[i_near] - quad_points_phys[i_quad]
                    Rdist = 0.
                    for i_coord in range(3):
                        Rdist += Rhat[i_coord] ** 2
                    Rdist = _np.sqrt(Rdist)
                    Rhat /= Rdist

                    term_M = _np.empty(3, dtype=_np.complex128)
                    scalar_prod = 0.
                    for i_coord in range(3):
                        scalar_prod += basis_eval[i_coord] * Rhat[i_coord]
                        term_M[i_coord] = (
                            basis_eval[(i_coord+1)%3] * Rhat[(i_coord-1)%3]
                            - basis_eval[(i_coord-1)%3] * Rhat[(i_coord+1)%3]
                        )
                    term_M *= -sol_E_bound[edge_m] * (1 + 1 / 1j / k0 / Rdist)

                    term_J = (1 - 1j / k0 / Rdist - 1 / (k0 * Rdist)**2) * basis_eval
                    term_J -= (1 - 3j / k0 / Rdist - 3 / (k0 * Rdist)**2) * scalar_prod * Rhat
                    term_J *= sol_H[edge_m]
                    E_sc[i_near] += (
                        (term_M + term_J) * _np.exp(-1j * k0 * Rdist) / 4 / _np.pi / Rdist
                        * quad_weights[i_quad]
                    ) * J_m

    E_sc *= -1j * k0


@_nb.jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def compute_far_field_integral(
    k0, edge2facet, facet_areas, quad_points, quad_weights, basis,
    unit_points, sol_E_bound, sol_H, Efar
):
    for i_point in _nb.prange(unit_points.shape[0]): # pylint: disable=not-an-iterable
        for edge_m in range(edge2facet.shape[0]):

            for i_facet_m in range(edge2facet.shape[1]):

                # Jacobian norm for integral m
                J_m = 2 * facet_areas[edge2facet[edge_m, i_facet_m]]

                # No multiplication by eta0 since the solH coeffs are already scaled by that
                for i_quad in range(quad_points.shape[1]):
                    basis_eval = basis[edge_m, i_facet_m, i_quad]
                    scalar_prod = 0.
                    cross_prod_M = _np.empty(3, dtype=_np.complex128)
                    cross_prod_J = _np.empty(3, dtype=_np.complex128)

                    for i_coord in range(3):
                        cross_prod_M[i_coord] = (
                            unit_points[i_point, (i_coord+1)%3] * basis_eval[(i_coord-1)%3]
                            - unit_points[i_point, (i_coord-1)%3] * basis_eval[(i_coord+1)%3]
                        )
                        scalar_prod += (
                            unit_points[i_point, i_coord]
                            * quad_points[edge2facet[edge_m, i_facet_m], i_quad, i_coord]
                        )

                    for i_coord in range(3):
                        cross_prod_J[i_coord] = (
                            unit_points[i_point, (i_coord+1)%3] * cross_prod_M[(i_coord-1)%3]
                            - unit_points[i_point, (i_coord-1)%3] * cross_prod_M[(i_coord+1)%3]
                        )

                    cross_prod_J *= sol_H[edge_m]
                    cross_prod_M *= -sol_E_bound[edge_m]

                    Efar[i_point] += (
                        (cross_prod_J + cross_prod_M) * _np.exp(1j * k0 * scalar_prod)
                        * quad_weights[i_quad] * J_m
                    )
