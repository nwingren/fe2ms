"""
Finite element boundary integral systems based on different solution approaches.
Currently supported approaches are:
    - Full equation system
    - Adaptive cross approximation (ACA) acceleration of boundary integral part

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

import os as _os
import pickle as _pickle
import logging

import numba as _nb

import numpy as _np
from scipy import linalg as _linalg
from scipy import sparse as _sparse
from scipy.sparse import linalg as _sparse_linalg

import dolfinx as _dolfinx

from scipy.constants import speed_of_light as _c0

from fe2ms.utility import (
    ComputationVolume as _ComputationVolume,
    FEBIBlocks as _FEBIBlocks,
    FEBISpaces as _FEBISpaces,
    connect_fe_bi_spaces as _connect_fe_bi_spaces
)
import fe2ms.assembly as _assembly
import fe2ms.preconditioners as _precs
import fe2ms.bi_space as _bi_space

LOGGER = logging.getLogger('febi')

class FEBISystem:
    """
    General FEBI system. For solution, use systems which inherit from this.
    """

    def __init__(
        self, frequency, computation_volume: _ComputationVolume,
        formulation: str
    ):
        """
        Initialize system by associating it to a frequency and computation volume.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.
        computation_volume : utility.ComputationVolume
            FEBI computation volume defining mesh, material and boundaries.
        formulation : {'is-efie', 'vs-efie', 'ej'}
            Formulation to use for finite element block structure and and
            boundary integral operators.
            'is-efie' uses I-S edge enumeration, i.e. explicit interior and surface finite element
            blocks, and uses the EFIE on the boundary.
            'vs-efie' uses V-S edge enumeration, i.e. one finite element block with transition
            operators to interior and surface, and uses the EFIE on the boundary.
            'ej' uses the symmetric formulation with both the EFIE and the MFIE. It requires use of
            I-S edge enumeration.
        """

        self.spaces = None

        # Solution vectors
        self.sol_E = None
        self.sol_H = None

        # Free space wavenumber
        self._k0 = 2 * _np.pi * frequency / _c0

        self.computation_volume = computation_volume
        if formulation not in ('is-efie', 'vs-efie', 'ej'):
            raise NotImplementedError(f'Formulation \'{formulation}\' not implemented')
        self._formulation = formulation
        self._source_fun = None
        self._rhs = None

        # System blocks with full BI matrices
        self._system_blocks = None

        # Preconditioner
        self.M_prec = None
        self.K_prec = None
        self.L_prec = None


    # TODO: Make this private and do automatically at system creation instead. There might have
    # been reason to have it like this before, but to the user it's only confusing like this
    def connect_fe_bi(self, save_file=None):
        """
        Connect FE and BI degrees of freedom with matrices T^SI and T^IS.

        Uses 1st order Nédélec basis functions for FE space and 1st order RWG functions tested with
        1st order SNC functions for BI space.

        Can save connection matrix for later use. If saved matrix is loaded it gives a faster
        runtime compared to building it from scratch.

        Parameters
        ----------
        save_file : str, optional
            Path and file name (including extension) for data to be loaded/saved, by default None.
            If the file exists with a correct identifier, connection data is loaded.
            If the file does not exist, connection data will be saved to it after generation.
        """

        identifier = 'FEBISystem-connection_matrix'
        load_data = False
        save_data = False

        # Check if save file exists and set logic accordingly
        if save_file is not None:
            if _os.path.isfile(save_file):
                load_data = True
            else:
                save_data = True

        # Load trace data. Only continue if data has the correct identifier
        if load_data:
            with open(save_file, 'rb') as inp:
                loaded = _pickle.load(inp)
                if not isinstance(loaded[0], str) or loaded[0] != identifier:
                    raise ValueError(
                        'Unable to load trace data from file - file does not '
                        'contain the correct data'
                    )
                T_SV, T_VS = loaded[1:]

        fe_space = _dolfinx.fem.FunctionSpace(
            self.computation_volume.mesh, ('N1curl', 1)
        )

        # Build connection matrix if needed
        if not load_data:
            T_SV, bi_meshdata = _connect_fe_bi_spaces(
                fe_space, self.computation_volume.get_external_facets()
            )
            T_VS = T_SV.copy().T
        else:
            ext_facets = self.computation_volume.get_external_facets()
            bi_mesh, fe_from_bi_facets_list = \
                _dolfinx.mesh.create_submesh(fe_space.mesh, 2, ext_facets)[:2]
            bi_meshdata = _bi_space.BIMeshData(
                bi_mesh, fe_space, fe_from_bi_facets_list,
                ext_facets
            )

        # Save trace data
        if save_data:
            with open(save_file, 'wb') as outp:
                _pickle.dump((identifier, T_SV, T_VS), outp)
            LOGGER.info('Connection matrix saved to %s', save_file)

        fe_size = fe_space.dofmap.index_map.size_global
        bi_size = bi_meshdata.edge2vert.shape[0]

        # Connection matrices for all FE DoFs <-> exclusively interior FE DoFs
        surf_dofs = T_SV.nonzero()[1]
        int_dofs = _np.setdiff1d(_np.arange(fe_size), surf_dofs, assume_unique=True)
        T_IV = _sparse.coo_array(
            (
                _np.ones(fe_size - bi_size),
                (_np.arange(fe_size - bi_size), int_dofs)
            ),
            shape=(fe_size - bi_size, fe_size), dtype="int64"
        ).tocsr()
        T_VI = T_IV.T

        bi_basisdata = _bi_space.BIBasisData(bi_meshdata)

        self.spaces = _FEBISpaces(
            fe_space, bi_meshdata, bi_basisdata, T_SV, T_VS, T_IV, T_VI, fe_size, bi_size
        )


    # TODO: Make a better system for incident waves. Need to be able to distinguish between single
    # and multiple incidence for example, as all postprocessing might not be appropriate for all
    # types of incidence (e.g. multiple plane wave incidence and rcs)
    # TODO: Add spherical coordinate version of this
    def set_source_plane_wave(self, amplitude, polarization, direction):
        """
        Set source function corresponding to an incident linearly polarized plane wave, or a linear
        combination of n linearly polarized plane waves.

        Parameters
        ----------
        amplitude : complex or ndarray
            Complex amplitude(s) of plane wave(s). Can be a complex scalar (single wave)
            or complex array of shape (n,) (linear combination of n waves).
        polarization : ndarray
            Vector for polarization(s) of plane wave(s).
            Can have shape (3,) (single wave) or (n,3) (linear combination of n waves).
            Will be normalized if it is not already unit vector(s).
        direction : ndarray
            Vector for propagation direction(s) of plane wave(s).
            Can have shape (3,) (single wave) or (n,3) (linear combination of n waves).
            Will be normalized if it is not already unit vector(s).
        """

        # Normalize vectors
        polarization = polarization.T / _linalg.norm(polarization.T, axis=0)
        direction = direction.T / _linalg.norm(direction.T, axis=0)

        polarization_H = _np.cross(direction.T, polarization.T).T

        if _np.isscalar(amplitude):
            self._source_fun = (
                lambda x: amplitude * polarization * _np.exp(-1j * self._k0 * x @ direction)[:, None],
                lambda x: amplitude * polarization_H * _np.exp(-1j * self._k0 * x @ direction)[:, None]
            )
        else:
            self._source_fun = (
                lambda x: _np.sum(
                    amplitude * polarization * _np.exp(-1j * self._k0 * x @ direction)[:, None],
                    axis=2
                ),
                lambda x: _np.sum(
                    amplitude * polarization_H * _np.exp(-1j * self._k0 * x @ direction)[:, None],
                    axis=2
                ),
            )

        b_inc = _assembly.assemble_rhs(
            self._formulation, self._k0, self.spaces.bi_meshdata,
            self.spaces.bi_basisdata, self._source_fun
        )

        if self._formulation in ('is-efie', 'vs-efie'):
            self._rhs = _np.concatenate((_np.zeros(self.spaces.fe_size), b_inc))
        elif self._formulation == 'ej':
            self._rhs = _np.concatenate((_np.zeros(self.spaces.fe_size - self.spaces.bi_size), b_inc))
    

    # TODO: Numba this!
    # TODO: Add cartesian coordinate version of this (and of rcs)
    def compute_far_field(
        self, r, theta, phi, total_field=False
    ):
        """
        Compute electric field in far-field region for multiple directions.
        Uses Volakis, Integral Equation Methods for Electromagnetics (2.77a).

        Parameters
        ----------
        r : float
            Radial distance.
        theta : ndarray
            Polar angles of directions.
        phi : ndarray
            Azimuth angles of direction.
        total_field : bool, by default False
            Whether to compute total field. By default, only the scattered field is computed.

        Returns
        -------
        Efar : ndarray
            Electric far-field.
        """

        if any(sol is None for sol in (self.sol_E, self.sol_H)):
            raise Exception('No solution computed!')

        unit_points = _np.stack(
            (_np.cos(phi)*_np.sin(theta),_np.sin(phi)*_np.sin(theta),_np.cos(theta)),
            axis=1
        )

        surf_integral = _np.zeros(unit_points.shape, dtype=_np.complex128)
        meshdata = self.spaces.bi_meshdata

        if self._formulation == 'vs-efie':
            sol_E_bound = self.spaces.T_SV @ self.sol_E
        else:
            sol_E_bound = self.sol_E[-self.spaces.bi_size:] # pylint: disable=unsubscriptable-object

        for edge_m in range(self.spaces.bi_size):

            for i_facet_m, facet_m in enumerate(meshdata.edge2facet[edge_m]):

                # Jacobian norm for integral m
                J_m = 2 * meshdata.facet_areas[facet_m]

                quad_points_phys = self.spaces.bi_basisdata.quad_points[facet_m]
                basis_eval = self.spaces.bi_basisdata.basis[edge_m, i_facet_m]

                # No multiplication by eta0 since the solH coeffs are already scaled by that
                rhat = _np.repeat(unit_points[None,:,:], quad_points_phys.shape[0], axis=0)
                scalar_product = _np.sum(rhat * quad_points_phys[:,None,:], axis=2, keepdims=True)
                cross_prod_M = _np.complex128(_np.cross(rhat, basis_eval[:,None,:]))
                cross_prod_J = _np.complex128(_np.cross(rhat, cross_prod_M))
                cross_prod_M *= -sol_E_bound[edge_m] # pylint: disable=unsubscriptable-object
                cross_prod_J *= self.sol_H[edge_m] # pylint: disable=unsubscriptable-object
                surf_integral += _np.sum(
                    (cross_prod_M + cross_prod_J) * self.spaces.bi_basisdata.quad_weights[:,None,None]
                    * _np.exp(1j * self._k0 * scalar_product),
                    axis=0
                ) * J_m

        Efar = 1j *self._k0 * _np.exp(-1j * self._k0 * r) / 4 / _np.pi / r * surf_integral

        if total_field:
            Efar += self._source_fun[0](r*unit_points)

        return Efar


    def compute_rcs(
        self, theta, phi
    ):
        """
        Compute bistatic RCS in multiple directions.
        Uses Volakis, Integral Equation Methods for Electromagnetics (2.77a).

        Parameters
        ----------
        theta : ndarray
            Polar angles of directions.
        phi : ndarray
            Azimuth angles of direction.

        Returns
        -------
        rcs : ndarray
            Bistatic rcs.
        """

        # FIXME: This amplitude does not work in general, only those waves with exp(jk.r)
        Efar_sc = self.compute_far_field(1., theta, phi)
        amplitude = _linalg.norm(self._source_fun[0](_np.array([[0., 0, 0]])))
        return 4 * _np.pi * _np.sum(_np.abs(Efar_sc)**2, axis=1) / amplitude**2


    # TODO: Implement near-field computation

    # TODO: Implement sources on the interior of entity?

class FEBISystemFull(FEBISystem):
    """
    FEBI system using the full block matrix structure.
    """


    def __init__(
        self, frequency, computation_volume,
        integral_equation
    ):

        super().__init__(frequency, computation_volume, integral_equation)

        # LU factorization object
        self._system_lufactor = None


    def assemble(self, compute_lu=False, save_file=None):
        """
        Assemble system as defined by computational volume with full block structure.
        Will build function spaces and connections if not already done. For direct solutions,
        an LU factorization can be pre-computed.

        Can save matrix data for later use. If saved matrices are loaded it gives a significantly
        faster runtime compared to assembly from scratch.

        Parameters
        ----------
        compute_lu : bool, optional
            Whether to compute LU factorization of system matrix after
            assembly, by default False.
        save_file : str, optional
            Path and file name (including extension) for data to be
            loaded/saved, by default None.
            If the file exists with a correct identifier, system blocks are loaded.
            If the file does not exist, system blocks will be saved to it after
            assembly.
        """

        identifier = 'FEBISystemFull-KBPQb_blocks'
        load_data = False
        save_data = False

        # Check if save file exists and set logic accordingly
        if save_file is not None:
            if _os.path.isfile(save_file):
                load_data = True
            else:
                save_data = True

        # Build function spaces if not already done
        if self.spaces is None:
            LOGGER.info('Connections not made, doing that first')
            self.connect_fe_bi()

        # Load matrix data, assemble otherwise. Only continue if data has the
        # correct identifier
        if load_data:
            with open(save_file, 'rb') as inp:
                loaded = _pickle.load(inp)
                if not isinstance(loaded[0], str) or loaded[0] != identifier:
                    raise ValueError(
                        'Unable to load matrix data from file - file does not '
                        'contain the correct data'
                    )
                K_matrix, B_matrix, P_matrix, Q_matrix = loaded[1:]
        else:
            K_matrix, B_matrix = _assembly.assemble_KB_blocks(
                self._k0, self.computation_volume,
                self.spaces
            )

            # Boundary integral matrices
            P_matrix, Q_matrix, self.K_prec, self.L_prec = _assembly.assemble_bi_blocks_full(
                self._formulation, self._k0, self.spaces.bi_meshdata, self.spaces.bi_basisdata
            )

            # Save block data
            if save_data:
                with open(save_file, 'wb') as outp:
                    _pickle.dump(
                        (identifier, K_matrix, B_matrix, P_matrix, Q_matrix), outp
                    )
                LOGGER.info('Matrix data saved to %s', save_file)

        # Compute LU factor if needed, otherwise store blocks
        if compute_lu:
            self._system_blocks = None
            if self._formulation in ('is-efie', 'ej'):
                K_II = self.spaces.T_IV @ K_matrix @ self.spaces.T_VI
                K_IS = self.spaces.T_IV @ K_matrix @ self.spaces.T_VS
                K_SI = (self.spaces.T_SV @ K_matrix) @ self.spaces.T_VI
                K_SS = self.spaces.T_SV @ K_matrix @ self.spaces.T_VS

            if self._formulation == 'is-efie':
                blocks = [
                    [K_II.toarray(), K_IS.toarray(), None],
                    [K_SI.toarray(), K_SS.toarray(), B_matrix.toarray()],
                    [None, P_matrix, Q_matrix]
                ]
            elif self._formulation == 'vs-efie':
                blocks = [
                    [K_matrix, self.spaces.T_VS @ B_matrix],
                    [P_matrix @ self.spaces.T_SV, Q_matrix]
                ]
            elif self._formulation == 'ej':
                blocks = [
                    [K_II, K_IS, None],
                    [K_SI, K_SS + 1j * self._k0 * Q_matrix, -1j * self._k0 * P_matrix],
                    [None, -1j * self._k0 * P_matrix.T, -1j * self._k0 * Q_matrix]
                ]
                blocks[1][1] = blocks[1][1].toarray()
            blocks[0][0] = blocks[0][0].toarray()
            blocks[0][1] = blocks[0][1].toarray()
            self._system_lufactor = _linalg.lu_factor(_np.block(blocks))
        else:
            self._system_blocks = _FEBIBlocks(K_matrix, B_matrix, P_matrix, Q_matrix)
            self._system_lufactor = None


    def solve(self):
        """
        Solve the system using default method
        (iterative solution w/ LGMRES and direct preconditioner).
        """

        self.solve_iterative()


    def solve_direct(self):
        """
        Solve system directly. Will use precomputed LU decomposition if it
        exists. Will assemble if not already done.

        Note that all zeros in sparse blocks will be explicitly stored
        if this solution method is used. Consider using solve_iterative or
        defining the system using FEBISystemInwardLooking instead.
        """

        if self._rhs is None:
            raise Exception('No right-hand-side set!')

        # Assemble system if not already done
        if self._system_blocks is None and self._system_lufactor is None:
            LOGGER.info('System not assembled, doing that before solving')
            self.assemble()

        # Solve using matrix or LU factor
        if self._system_blocks is None:
            sol = _linalg.lu_solve(self._system_lufactor, self._rhs)
        else:
            if self._formulation in ('is-efie', 'ej'):
                K_II = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VI
                K_IS = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VS
                K_SI = (self.spaces.T_SV @ self._system_blocks.K) @ self.spaces.T_VI
                K_SS = self.spaces.T_SV @ self._system_blocks.K @ self.spaces.T_VS

            if self._formulation == 'is-efie':
                blocks = [
                    [K_II.toarray(), K_IS.toarray(), None],
                    [K_SI.toarray(), K_SS.toarray(), self._system_blocks.B.toarray()],
                    [None, self._system_blocks.P, self._system_blocks.Q]
                ]
            elif self._formulation == 'vs-efie':
                blocks = [
                    [self._system_blocks.K, self.spaces.T_VS @ self._system_blocks.B],
                    [self._system_blocks.P @ self.spaces.T_SV, self._system_blocks.Q]
                ]
            elif self._formulation == 'ej':
                blocks = [
                    [K_II, K_IS, None],
                    [K_SI, K_SS + 1j * self._k0 * self._system_blocks.Q, -1j * self._k0 * self._system_blocks.P],
                    [None, -1j * self._k0 * self._system_blocks.P.T, -1j * self._k0 * self._system_blocks.Q]
                ]
                blocks[1][1] = blocks[1][1].toarray()
            blocks[0][0] = blocks[0][0].toarray()
            blocks[0][1] = blocks[0][1].toarray()
            sol = _linalg.solve(_np.block(blocks), self._rhs)

        self.sol_E = sol[:self.spaces.fe_size]
        self.sol_H = sol[self.spaces.fe_size:]


    def solve_iterative(
        self, solver=None, preconditioner=None, new_prec=False,
        right_prec=True, return_prec=False, counter=None, solver_tol=1e-3
    ):
        """
        Solve system iteratively with preconditioning. Since the FE-BI system
        is highly ill-conditioned, a preconditioner is necessary for almost all
        problems. Will assemble if not already done.

        Parameters
        ----------
        solver : function, optional
            Iterative solver to use, by default scipy.sparse.linalg.lgmres. This should have the
            same call structure as other iterative solvers in scipy.sparse.linalg.
        preconditioner : function, optional
            Preconditioner to use, with the default being febicode.preconditioners.direct.
            Preconditioners can be generated by functions in febicode.preconditioners, or created
            manually. A manually created preconditioner should be a function which takes a solution
            vector as its argument and returns the preconditioned vector.
        new_prec : bool, optional
            Whether to generate a new preconditioner even if there is an existing one already,
            by default False.
        right_prec : bool, optional
            Whether to use right preconditioning instead of left, by default True.
        return_prec : bool, optional
            Whether to return the preconditioner LinearOperator, by default False.
        counter : function, optional
            Callback function to scipy iterative solver, by default None. Note that different
            solvers call this differently. For more info, see scipy documentation.
        solver_tol : float, optional
            (Relative) tolerance of iterative solver, by default 1e-3.

        Returns
        -------
        info : int
            0 if solver converged, >0 if solver did not converge,
            >0 if illegal input or breakdown of solver.
        M_prec : function | NoneType
            Preconditioner function which applies preconditioner in matrix-vector-products.
            Is None if return_prec == False.
        """

        if self._rhs is None:
            raise Exception('No right-hand-side set!')

        if self._system_blocks is None:
            if self._system_lufactor is not None:
                LOGGER.info('LU factorization not allowed, performing assembly again')
            else:
                LOGGER.info('System not assembled, doing that before solving')
            self.assemble()

        if solver is None:
            solver = _sparse_linalg.lgmres

        # Generate or load preconditioner
        if preconditioner is None:
            if self.M_prec is None or new_prec:
                self.M_prec = _precs.direct(self)
        else:
            self.M_prec = preconditioner

        # LinearOperator for matrix-vector multiplication such that sparse blocks are kept that way
        if self._formulation in ('is-efie', 'ej'):
            K_II = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VI
            K_IS = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VS
            K_SI = (self.spaces.T_SV @ self._system_blocks.K) @ self.spaces.T_VI
            K_SS = self.spaces.T_SV @ self._system_blocks.K @ self.spaces.T_VS

        fe_size = self.spaces.fe_size
        bi_size = self.spaces.bi_size
        in_size = fe_size - bi_size
        
        if self._formulation == 'is-efie':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        self._system_blocks.P @ x[in_size:-bi_size]
                        + self._system_blocks.Q @ x[-bi_size:]
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        self._system_blocks.P @ x[in_size:-bi_size]
                        + self._system_blocks.Q @ x[-bi_size:]
                    ))
        elif self._formulation == 'vs-efie':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + self._system_blocks.Q @ x[-bi_size:]
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + self._system_blocks.Q @ x[-bi_size:]
                    ))
        elif self._formulation == 'ej':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + 1j * self._k0 * self._system_blocks.Q @ x[in_size:-bi_size]
                        - 1j * self._k0 * self._system_blocks.P @ x[-bi_size:],
                        - 1j * self._k0 * self._system_blocks.P.T @ x[in_size:-bi_size]
                        - 1j * self._k0 * self._system_blocks.Q @ x[-bi_size:]
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + 1j * self._k0 * self._system_blocks.Q @ x[in_size:-bi_size]
                        - 1j * self._k0 * self._system_blocks.P @ x[-bi_size:],
                        - 1j * self._k0 * self._system_blocks.P.T @ x[in_size:-bi_size]
                        - 1j * self._k0 * self._system_blocks.Q @ x[-bi_size:]
                    ))

        system_operator = _sparse_linalg.LinearOperator(
            shape = 2 * (fe_size + bi_size,),
            matvec = matvec_fun,
            dtype = _np.complex128
        )

        if counter is None:
            if right_prec:
                sol, info = solver(system_operator, self._rhs, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(system_operator, self._rhs, M=M_op, tol=solver_tol)
        else:
            if right_prec:
                sol, info = solver(system_operator, self._rhs, callback=counter, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(
                    system_operator, self._rhs, M=M_op, callback=counter, tol=solver_tol
                )
        
        self.sol_E = sol[:self.spaces.fe_size]
        self.sol_H = sol[self.spaces.fe_size:]

        if return_prec:
            return info, self.M_prec
        else:
            return info, None


class FEBISystemACA(FEBISystem):
    """
    FEBI system where the BI blocks are assembled using the adaptive cross approximation (ACA).
    """

    def __init__(self, frequency, computation_volume: _ComputationVolume, integral_equation: str):

        super().__init__(frequency, computation_volume, integral_equation)

        # Far interactions used in solution
        self._far_operator = None


    def assemble(
        self, target_points=None, recompress=True, tolerance=1e-3
    ):
        """
        Assemble system using a multilevel ACA for the BI blocks. An octree is generated for this
        with target_points DoFs per group at the lowest level. The octree is generated iteratively
        to arrive at a tree with a mean value for the DoFs per group close to this.

        Parameters
        ----------
        target_points : int, optional
            Target for the mean number of points per group, by default None. The default will set
            this to the square root of the number of BI DoFs.
        tolerance : float, optional
            Error tolerance determining termination of the ACA, by default 1e-3.
        recompress : bool, optional
            Whether to recompress ACA matrices using QR+SVD using same tolerance, dy default True.
        """

        # Build function spaces if not already done
        if self.spaces is None:
            LOGGER.info('Connections not made, doing that first')
            self.connect_fe_bi()

        K_matrix, B_matrix = _assembly.assemble_KB_blocks(
            self._k0, self.computation_volume,
            self.spaces
        )

        # Boundary integral blocks
        P_near, Q_near, far_operator, self.K_prec, self.L_prec = _assembly.assemble_bi_aca(
            self._formulation, self._k0, self.spaces.bi_meshdata,
            self.spaces.bi_basisdata, recompress=recompress, target_points=target_points,
            tolerance=tolerance
        )

        self._system_blocks = _FEBIBlocks(K_matrix, B_matrix, P_near, Q_near)
        self._far_operator = far_operator


    def solve(self):
        """
        Solve the system using default method
        (iterative solution w/ LGMRES and direct preconditioner).
        """
        self.solve_iterative()


    def solve_iterative(
        self, solver=None, preconditioner=None, new_prec=False,
        right_prec=True, return_prec=False, counter=None, solver_tol=1e-3
    ):
        """
        Solve system iteratively with preconditioning. Since the FE-BI system is highly
        ill-conditioned, a preconditioner is necessary for almost all problems.
        Will assemble if not already done.

        Parameters
        ----------
        solver : function, optional
            Iterative solver to use, by default scipy.sparse.linalg.lgmres. This should have the
            same call structure as other iterative solvers in scipy.sparse.linalg.
        preconditioner : function, optional
            Preconditioner to use, with the default being febicode.preconditioners.direct.
            Preconditioners can be generated by functions in febicode.preconditioners, or created
            manually. A manually created preconditioner should be a function which takes a solution
            vector as its argument and returns the preconditioned vector.
        new_prec : bool, optional
            Whether to generate a new preconditioner even if there is an existing one already,
            by default False.
        right_prec : bool, optional
            Whether to use right preconditioning instead of left, by default True.
        return_prec : bool, optional
            Whether to return the preconditioner LinearOperator, by default False.
        counter : function, optional
            Callback function to scipy iterative solver, by default None. Note that different
            solvers call this differently. For more info, see scipy documentation.
        solver_tol : float, optional
            (Relative) tolerance of iterative solver, by default 1e-3.

        Returns
        -------
        info : int
            0 if solver converged, >0 if solver did not converge,
            >0 if illegal input or breakdown of solver.
        M_prec : LinearOperator | NoneType
            Preconditioner object. Either a sparse matrix or a linear operator which applies
            preconditioner in matrix-vector-products. Is None if return_prec == False.
        """

        if self._rhs is None:
            raise Exception('No right-hand-side set!')

        if self._system_blocks is None:
            LOGGER.info('System not assembled, doing that before solving')
            self.assemble()
        
        if solver is None:
            solver = _sparse_linalg.lgmres

        # Generate or load preconditioner
        if preconditioner is None:
            if self.M_prec is None or new_prec:
                self.M_prec = _precs.direct(self)
        else:
            self.M_prec = preconditioner

        # LinearOperator for matrix-vector multiplication such that sparse blocks are kept that way
        if self._formulation in ('is-efie', 'ej'):
            K_II = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VI
            K_IS = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VS
            K_SI = (self.spaces.T_SV @ self._system_blocks.K) @ self.spaces.T_VI
            K_SS = self.spaces.T_SV @ self._system_blocks.K @ self.spaces.T_VS

        fe_size = self.spaces.fe_size
        bi_size = self.spaces.bi_size
        in_size = fe_size - bi_size
        
        if self._formulation == 'is-efie':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        self._system_blocks.P @ x[in_size:-bi_size]
                        + self._far_operator.matvec_Kop(x[in_size:-bi_size])
                        + self._system_blocks.Q @ x[-bi_size:]
                        + self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        self._system_blocks.P @ x[in_size:-bi_size]
                        + self._far_operator.matvec_Kop(x[in_size:-bi_size])
                        + self._system_blocks.Q @ x[-bi_size:]
                        + self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
        elif self._formulation == 'vs-efie':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + self._far_operator.matvec_Kop(self.spaces.T_SV @ x[:fe_size])
                        + self._system_blocks.Q @ x[-bi_size:]
                        + self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + self._far_operator.matvec_Kop(self.spaces.T_SV @ x[:fe_size])
                        + self._system_blocks.Q @ x[-bi_size:]
                        + self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
        elif self._formulation == 'ej':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + 1j * self._k0 * (
                            self._system_blocks.Q @ x[in_size:-bi_size]
                            + self._far_operator.matvec_Lop(x[in_size:-bi_size])
                        )
                        - 1j * self._k0 * (
                            self._system_blocks.P @ x[-bi_size:]
                            + self._far_operator.matvec_Kop(x[-bi_size:])
                        ),
                        - 1j * self._k0 * (
                            self._system_blocks.P.T @ x[in_size:-bi_size]
                            + self._far_operator.matvec_Kop_T(x[in_size:-bi_size])
                        )
                        - 1j * self._k0 * (
                            self._system_blocks.Q @ x[-bi_size:]
                            + self._far_operator.matvec_Lop(x[fe_size:])
                        )
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + 1j * self._k0 * (
                            self._system_blocks.Q @ x[in_size:-bi_size]
                            + self._far_operator.matvec_Lop(x[in_size:-bi_size])
                        )
                        - 1j * self._k0 * (
                            self._system_blocks.P @ x[-bi_size:]
                            + self._far_operator.matvec_Kop(x[-bi_size:])
                        ),
                        - 1j * self._k0 * (
                            self._system_blocks.P.T @ x[in_size:-bi_size]
                            + self._far_operator.matvec_Kop_T(x[in_size:-bi_size])
                        )
                        - 1j * self._k0 * (
                            self._system_blocks.Q @ x[-bi_size:]
                            + self._far_operator.matvec_Lop(x[fe_size:])
                        )
                    ))

        system_operator = _sparse_linalg.LinearOperator(
            shape = 2 * (self.spaces.fe_size + self.spaces.bi_size,),
            matvec = matvec_fun,
            dtype = _np.complex128
        )

        if counter is None:
            if right_prec:
                sol, info = solver(system_operator, self._rhs, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(system_operator, self._rhs, M=M_op, tol=solver_tol)
        else:
            if right_prec:
                sol, info = solver(system_operator, self._rhs, callback=counter, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(
                    system_operator, self._rhs, M=M_op, callback=counter, tol=solver_tol
                )
        
        self.sol_E = sol[:self.spaces.fe_size]
        self.sol_H = sol[self.spaces.fe_size:]

        if return_prec:
            return info, self.M_prec
        else:
            return info, None
