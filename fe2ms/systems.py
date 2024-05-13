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
Finite element boundary integral systems based on different solution approaches.

Currently supported approaches are:
    - Full equation system
    - Adaptive cross approximation (ACA) acceleration of boundary integral part
"""

import os as _os
import pickle as _pickle
import logging

import numpy as _np
from scipy import linalg as _linalg
from scipy import sparse as _sparse
from scipy.sparse import linalg as _sparse_linalg
from scipy.constants import speed_of_light as _c0, epsilon_0 as _eps0, mu_0 as _mu0

import dolfinx as _dolfinx

from fe2ms.utility import (
    ComputationVolume as _ComputationVolume,
    FEBIBlocks as _FEBIBlocks,
    FEBISpaces as _FEBISpaces,
    connect_fe_bi_spaces as _connect_fe_bi_spaces
)
import fe2ms.assembly as _assembly
import fe2ms.result_computations as _result_computations
import fe2ms.preconditioners as _precs
import fe2ms.bi_space as _bi_space

LOGGER = logging.getLogger('febi')

class FEBISystem:
    """
    General FEBI system. For solution, use systems which inherit from this.
    """

    def __init__(
        self, frequency, computation_volume: _ComputationVolume,
        formulation='is-efie'
    ):
        """
        Initialize system by associating it to a frequency and computation volume.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.
        computation_volume : utility.ComputationVolume
            FEBI computation volume defining mesh, material and boundaries.
        formulation : {'is-efie', 'vs-efie', 'ej', 'teth'}, optional.
            Formulation to use for finite element block structure and and
            boundary integral operators, by default 'is-efie'

            'is-efie' uses I-S edge enumeration, i.e. explicit interior and surface finite element
            blocks, and uses the EFIE on the boundary.

            'vs-efie' uses V-S edge enumeration, i.e. one finite element block with transition
            operators to interior and surface, and uses the EFIE on the boundary.

            'ej' uses the symmetric formulation with both the EFIE and the MFIE. It requires use of
            I-S edge enumeration.
            
            'teth' is a CFIE formulation which uses I-S edge enumeration. It reduces, but does not
            eliminate, the influence of interior resonances.
        """

        self.spaces = None

        # Solution vectors
        self.sol_E = None
        self.sol_H = None

        # Free space wavenumber
        self._k0 = 2 * _np.pi * frequency / _c0

        self.computation_volume = computation_volume
        if formulation not in ('is-efie', 'vs-efie', 'ej', 'teth'):
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
    def connect_fe_bi(self, save_file=None, quad_order=2):
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
        quad_order : int, optional
            Order of quadrature used in BI integrals, by default 2. It is suggested that order 4 is
            used instead of 3 for a higher order quadrature as they use the same number of points.
            See basix.make_quadrature(basix.CellType.triangle, quad_order) for more details on
            exact quadrature points and weights.
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

        fe_space = _dolfinx.fem.functionspace(
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

        bi_basisdata = _bi_space.BIBasisData(bi_meshdata, quad_order=quad_order)

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

        if self._formulation in ('is-efie', 'vs-efie', 'teth'):
            self._rhs = _np.concatenate((_np.zeros(self.spaces.fe_size), b_inc))
        elif self._formulation == 'ej':
            self._rhs = _np.concatenate(
                (_np.zeros(self.spaces.fe_size - self.spaces.bi_size), b_inc)
            )


    # TODO: Test this!
    def set_source_dipole(self, dipole_moment, polarization, origin):
        """
        Set source function corresponding to a Hertzian dipole.

        Parameters
        ----------
        dipole_moment : complex
            Dipole moment (current * length) of the dipole.
        polarization : ndarray
            Vector of shape (3,) for polarization of the dipole in cartesian coordinates.
            Will be normalized if it is not already unit vector.
        origin : ndarray
            Vector of shape (3,) for origin of the dipole in cartesian coordinates.
        """

        polarization /= _linalg.norm(polarization)
        phi_p = _np.arctan2(polarization[1], polarization[0])
        theta_p = _np.arccos(polarization[2])
        eta0 = _np.sqrt(_mu0 / _eps0)
        
        def Ed(x):
            x -= origin
            r = _np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
            phi = _np.arctan2(x[:,1], x[:,0]) - phi_p
            theta = _np.arccos(x[:,2]/r) - theta_p
            r_hat = _np.array([
                _np.sin(theta) * _np.cos(phi),
                _np.sin(theta) * _np.sin(phi),
                _np.cos(theta)
            ])
            theta_hat = _np.array([
                _np.cos(theta) * _np.cos(phi),
                _np.cos(theta) * _np.sin(phi),
                -_np.sin(theta)
            ])
            return (
                r_hat * eta0 * dipole_moment * _np.cos(theta) / (2 * _np.pi * r**2)
                * (1 + 1 / (1j * self._k0 * r)) * _np.exp(-1j * self._k0 * r)
                + theta_hat * 1j * self._k0 * eta0 * dipole_moment * _np.sin(theta) / (4 * _np.pi * r)
                * (1 + 1 / (1j * self._k0 * r) - 1 / (self._k0 * r)**2) * _np.exp(-1j * self._k0 * r)
            ).T
        
        def Hd_bar(x):
            x -= origin
            r = _np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
            phi = _np.arctan2(x[:,1], x[:,0]) - phi_p
            theta = _np.arccos(x[:,2]/r) - theta_p
            phi_hat = _np.array([
                -_np.sin(phi),
                _np.cos(phi),
                _np.zeros_like(phi)
            ])
            return (
                phi_hat * 1j * self._k0 * dipole_moment * _np.sin(theta) / (4 * _np.pi * r)
                * (1 + 1 / (1j * self._k0 * r)) * _np.exp(-1j * self._k0 * r) * eta0
            ).T
        
        self._source_fun = (Ed, Hd_bar)

        b_inc = _assembly.assemble_rhs(
            self._formulation, self._k0, self.spaces.bi_meshdata,
            self.spaces.bi_basisdata, self._source_fun
        )

        if self._formulation in ('is-efie', 'vs-efie', 'teth'):
            self._rhs = _np.concatenate((_np.zeros(self.spaces.fe_size), b_inc))
        elif self._formulation == 'ej':
            self._rhs = _np.concatenate(
                (_np.zeros(self.spaces.fe_size - self.spaces.bi_size), b_inc)
            )


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

        meshdata = self.spaces.bi_meshdata
        basisdata = self.spaces.bi_basisdata

        if self._formulation == 'vs-efie':
            sol_E_bound = self.spaces.T_SV @ self.sol_E
        else:
            sol_E_bound = self.sol_E[-self.spaces.bi_size:] # pylint: disable=unsubscriptable-object

        Efar = _np.zeros_like(unit_points, dtype=_np.complex128)
        _result_computations.compute_far_field_integral(
            self._k0, meshdata.edge2facet, meshdata.facet_areas, basisdata.quad_points,
            basisdata.quad_weights, basisdata.basis, unit_points, sol_E_bound, self.sol_H, Efar
        )
        Efar *= 1j *self._k0 * _np.exp(-1j * self._k0 * r) / 4 / _np.pi / r

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


    def compute_near_field(self, x, y, z):
        """
        Compute electric near-field in a set of points. Computations are done both in FE and BI
        regions. Note that field values near the boundary between these regions are inaccurate!

        Parameters
        ----------
        x : ndarray
            x-coordinates of evaluation points.
        y : nddarray
            y-coordinates of evaluation points.
        z : ndarray
            z-coordinates of evaluation points.
        
        Returns
        -------
        efield_near : ndarray
            Electric near-field.
        """

        if any(sol is None for sol in (self.sol_E, self.sol_H)):
            raise Exception('No solution computed!')

        points_near = _np.stack((x, y, z), axis=1)
        efield_near = _np.zeros_like(points_near, dtype=_np.complex128)

        # Trees for finding dolfinx geoemtry
        bb_tree = _dolfinx.geometry.BoundingBoxTree(self.spaces.fe_space.mesh, 3)
        midpoint_tree = _dolfinx.geometry.create_midpoint_tree(
            self.spaces.fe_space.mesh, 3,
            _np.arange(self.spaces.fe_space.mesh.geometry.dofmap.num_nodes, dtype=_np.int32)
        )

        # Find which points are in FE and BI regions
        coll = _dolfinx.geometry.compute_collisions(bb_tree, points_near)
        coll_cells = _dolfinx.geometry.compute_colliding_cells(
            self.spaces.fe_space.mesh, coll, points_near
        )
        fe_points = _np.full(points_near.shape[0], False)
        bi_points = _np.full(points_near.shape[0], True)
        for i in range(coll_cells.num_nodes):
            if coll_cells.links(i).size > 0:
                fe_points[i] = True
                bi_points[i] = False

        efield_fun_fe = _dolfinx.fem.Function(self.spaces.fe_space)
        if self._formulation == 'vs-efie':
            efield_fun_fe.vector[:] = self.sol_E
        else:
            in_size = self.spaces.fe_size - self.spaces.bi_size
            vec_vs = _np.zeros_like(self.sol_E)
            vec_vs += self.spaces.T_VI @ self.sol_E[:in_size] # pylint: disable=unsubscriptable-object
            vec_vs += self.spaces.T_VS @ self.sol_E[in_size:] # pylint: disable=unsubscriptable-object
            efield_fun_fe.vector[:] = vec_vs
        closest_cells = _dolfinx.geometry.compute_closest_entity(
            bb_tree, midpoint_tree, self.spaces.fe_space.mesh, points_near[fe_points]
        )

        # Interpolate into Lagrange for proper visualization
        V0 = _dolfinx.fem.VectorFunctionSpace(self.spaces.fe_space.mesh, ('DG', 1))
        efield_fun_cg = _dolfinx.fem.Function(V0, dtype=_np.complex128)
        efield_fun_cg.interpolate(efield_fun_fe)
        efield_near[fe_points] = efield_fun_cg.eval(points_near[fe_points], closest_cells)

        # Compute scattered near field in BI region
        if self._formulation == 'vs-efie':
            sol_E_bound = self.spaces.T_SV @ self.sol_E
        else:
            sol_E_bound = self.sol_E[-self.spaces.bi_size:] # pylint: disable=unsubscriptable-object
        E_sc = _np.zeros_like(points_near[bi_points], dtype=_np.complex128)
        _result_computations.compute_near_field_bi(
            self._k0, self.spaces.bi_meshdata.edge2facet, self.spaces.bi_meshdata.facet_areas,
            self.spaces.bi_basisdata.quad_points, self.spaces.bi_basisdata.quad_weights,
            self.spaces.bi_basisdata.basis,
            points_near[bi_points], sol_E_bound, self.sol_H, E_sc
        )
        efield_near[bi_points] = E_sc + self._source_fun[0](points_near[bi_points])

        return efield_near

    # TODO: Implement sources on the interior of entity?

class FEBISystemFull(FEBISystem):
    """
    FEBI system using the full block matrix structure.
    """


    def __init__(
        self, frequency, computation_volume,
        formulation=None
    ):

        if formulation is None:
            super().__init__(frequency, computation_volume)
        else:
            super().__init__(frequency, computation_volume, formulation)

        # LU factorization object
        self._system_lufactor = None


    def assemble(self, compute_lu=False, save_file=None, quad_order_singular=5):
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
        quad_order_singular : int, optional
            Quadrature order to use in DEMCEM singular integral computations, by default 5.
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
                self._k0, self.spaces.bi_meshdata, self.spaces.bi_basisdata, quad_order_singular
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
            if self._formulation != 'vs-efie':
                K_II = self.spaces.T_IV @ K_matrix @ self.spaces.T_VI
                K_IS = self.spaces.T_IV @ K_matrix @ self.spaces.T_VS
                K_SI = (self.spaces.T_SV @ K_matrix) @ self.spaces.T_VI
                K_SS = self.spaces.T_SV @ K_matrix @ self.spaces.T_VS

            if self._formulation == 'is-efie':
                blocks = [
                    [K_II, K_IS, _np.zeros(K_IS.shape)],
                    [K_SI.toarray(), K_SS.toarray(), B_matrix.toarray()],
                    [_np.zeros(K_SI.shape), P_matrix, Q_matrix]
                ]
            elif self._formulation == 'vs-efie':
                blocks = [
                    [K_matrix, self.spaces.T_VS @ B_matrix],
                    [P_matrix @ self.spaces.T_SV, Q_matrix]
                ]
            elif self._formulation == 'ej':
                eta0 = _np.sqrt(_mu0 / _eps0)
                blocks = [
                    [K_II, K_IS, _np.zeros(K_IS.shape)],
                    [K_SI.toarray(), K_SS + 1j * self._k0 * Q_matrix, -1j * self._k0 * eta0 * P_matrix.T],
                    [_np.zeros(K_SI.shape), -1j * self._k0 * eta0 * P_matrix, -1j * self._k0 * eta0**2 * Q_matrix]
                ]
            elif self._formulation == 'teth':
                blocks = [
                    [K_II, K_IS, _np.zeros(K_IS.shape)],
                    [K_SI.toarray(), K_SS.toarray(), B_matrix.toarray()],
                    [_np.zeros(K_SI.shape), 0.5 * (P_matrix - Q_matrix), 0.5 * (P_matrix + Q_matrix)]
                ]
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
            if self._formulation != 'vs-efie':
                K_II = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VI
                K_IS = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VS
                K_SI = (self.spaces.T_SV @ self._system_blocks.K) @ self.spaces.T_VI
                K_SS = self.spaces.T_SV @ self._system_blocks.K @ self.spaces.T_VS

            if self._formulation == 'is-efie':
                blocks = [
                    [K_II, K_IS, _np.zeros(K_IS.shape)],
                    [K_SI.toarray(), K_SS.toarray(), self._system_blocks.B.toarray()],
                    [_np.zeros(K_SI.shape), self._system_blocks.P, self._system_blocks.Q]
                ]
            elif self._formulation == 'vs-efie':
                blocks = [
                    [self._system_blocks.K, self.spaces.T_VS @ self._system_blocks.B],
                    [self._system_blocks.P @ self.spaces.T_SV, self._system_blocks.Q]
                ]
            elif self._formulation == 'ej':
                eta0 = _np.sqrt(_mu0 / _eps0)
                blocks = [
                    [K_II, K_IS, _np.zeros(K_IS.shape)],
                    [
                        K_SI.toarray(),
                        K_SS + 1j * self._k0 * self._system_blocks.Q,
                        -1j * self._k0 * eta0 * self._system_blocks.P.T
                    ],
                    [
                        _np.zeros(K_SI.shape),
                        -1j * self._k0 * eta0 * self._system_blocks.P,
                        -1j * self._k0 * eta0**2 * self._system_blocks.Q
                    ]
                ]
            elif self._formulation == 'teth':
                blocks = [
                    [K_II, K_IS, _np.zeros(K_IS.shape)],
                    [K_SI.toarray(), K_SS.toarray(), self._system_blocks.B.toarray()],
                    [
                        _np.zeros(K_SI.shape),
                        0.5 * (self._system_blocks.P - self._system_blocks.Q),
                        0.5 * (self._system_blocks.P + self._system_blocks.Q)
                    ]
                ]
            blocks[0][0] = blocks[0][0].toarray()
            blocks[0][1] = blocks[0][1].toarray()
            sol = _linalg.solve(_np.block(blocks), self._rhs)

        self.sol_E = sol[:self.spaces.fe_size]
        self.sol_H = sol[self.spaces.fe_size:]

        if self._formulation == 'ej':
            self.sol_H *= _np.sqrt(_mu0 / _eps0)


    def solve_iterative(
        self, solver=None, preconditioner=None, new_prec=False, right_prec=True, counter=None,
        solver_tol=1e-5, bi_scale=1.
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
            Preconditioner to use, with the default being fe2ms.preconditioners.direct.
            Preconditioners can be generated by functions in fe2ms.preconditioners, or created
            manually. A manually created preconditioner should be a function which takes a solution
            vector as its argument and returns the preconditioned vector.
        new_prec : bool, optional
            Whether to generate a new preconditioner even if there is an existing one already,
            by default False.
        right_prec : bool, optional
            Whether to use right preconditioning instead of left, by default True.
        counter : function, optional
            Callback function to scipy iterative solver, by default None. Note that different
            solvers call this differently. For more info, see scipy documentation.
        solver_tol : float, optional
            (Relative) tolerance of iterative solver, by default 1e-5.
        bi_scale : complex, optional
            Scaling factor for the BI equation, by default 1. Not applied to EJ formulation.

        Returns
        -------
        info : int
            0 if solver converged, >0 if solver did not converge,
            <0 if illegal input or breakdown of solver.
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

        if self._formulation == 'ej':
            bi_scale = 1.

        # LinearOperator for matrix-vector multiplication such that sparse blocks are kept that way
        if self._formulation != 'vs-efie':
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
                        bi_scale * self._system_blocks.P @ x[in_size:-bi_size]
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        bi_scale * self._system_blocks.P @ x[in_size:-bi_size]
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                    ))
        elif self._formulation == 'vs-efie':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        bi_scale * self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        bi_scale * self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                    ))
        elif self._formulation == 'ej':
            eta0 = _np.sqrt(_mu0 / _eps0)
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + 1j * self._k0 * self._system_blocks.Q @ x[in_size:-bi_size]
                        - 1j * self._k0 * eta0 * self._system_blocks.P.T @ x[-bi_size:],
                        - 1j * self._k0 * eta0 * self._system_blocks.P @ x[in_size:-bi_size]
                        - 1j * self._k0 * eta0**2 * self._system_blocks.Q @ x[-bi_size:]
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + 1j * self._k0 * self._system_blocks.Q @ x[in_size:-bi_size]
                        - 1j * self._k0 * eta0 * self._system_blocks.P.T @ x[-bi_size:],
                        - 1j * self._k0 * eta0 * self._system_blocks.P @ x[in_size:-bi_size]
                        - 1j * self._k0 * eta0**2 * self._system_blocks.Q @ x[-bi_size:]
                    ))
        elif self._formulation == 'teth':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        bi_scale * 0.5 * (
                            self._system_blocks.P @ x[in_size:-bi_size]
                            - self._system_blocks.Q @ x[in_size:-bi_size]
                        )
                        + bi_scale * 0.5 * (
                            self._system_blocks.P @ x[-bi_size:]
                            + self._system_blocks.Q @ x[-bi_size:]
                        )
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        bi_scale * 0.5 * (
                            self._system_blocks.P @ x[in_size:-bi_size]
                            - self._system_blocks.Q @ x[in_size:-bi_size]
                        ) 
                        + bi_scale * 0.5 * (
                            self._system_blocks.P @ x[-bi_size:]
                            + self._system_blocks.Q @ x[-bi_size:]
                        )
                    ))

        system_operator = _sparse_linalg.LinearOperator(
            shape = 2 * (fe_size + bi_size,),
            matvec = matvec_fun,
            dtype = _np.complex128
        )

        if counter is None:
            if right_prec:
                sol, info = solver(system_operator, bi_scale * self._rhs, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(system_operator, bi_scale * self._rhs, M=M_op, tol=solver_tol)
        else:
            if right_prec:
                sol, info = solver(system_operator, bi_scale * self._rhs, callback=counter, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(
                    system_operator, bi_scale * self._rhs, M=M_op, callback=counter, tol=solver_tol
                )

        self.sol_E = sol[:self.spaces.fe_size]
        self.sol_H = sol[self.spaces.fe_size:]

        if self._formulation == 'ej':
            self.sol_H *= _np.sqrt(_mu0 / _eps0)

        return info
    

    def solve_semidirect(
        self, solver=None, counter=None, solver_tol=1e-5, lu_solve=None
    ):
        """
        Solve system by first eliminating the FE part using a sparse LU factorization, and then
        using this with the BI part in an iterative solver.

        No preconditioning options at this time. Not available for 'ej' formulation.

        This method is suitable for problems with large FE part and small BI part, or  particularly
        ill-conditioned problems with small enough BI part.
        Will assemble if not already done.

        Parameters
        ----------
        solver : function, optional
            Iterative solver to use, by default scipy.sparse.linalg.lgmres. This should have the
            same call structure as other iterative solvers in scipy.sparse.linalg.
        counter : function, optional
            Callback function to scipy iterative solver, by default None. Note that different
            solvers call this differently. For more info, see scipy documentation.
        solver_tol : float, optional
            (Relative) tolerance of iterative solver, by default 1e-5.
        lu_solve : function, optional
            Function corresponding to 'solve' operation for the LU factorization of the FE part.
            This is used instead of computing new factorizations using the system matrix.

        Returns
        -------
        info : int
            0 if solver converged, >0 if solver did not converge,
            <0 if illegal input or breakdown of solver.
        lu_solve : function
            Function corresponding to 'solve' operation for the LU factorization of the FE part
            which was used in the solution
        """

        if self._rhs is None:
            raise Exception('No right-hand-side set!')

        if self._system_blocks is None:
            LOGGER.info('System not assembled, doing that before solving')
            self.assemble()
        
        if solver is None:
            solver = _sparse_linalg.lgmres

        if self._formulation == 'ej':
            raise Exception('Semidirect solution not available for ej formulation')
        
        fe_size = self.spaces.fe_size
        bi_size = self.spaces.bi_size
        in_size = fe_size - bi_size
        
        if lu_solve is None:
            if self._formulation != 'vs-efie':
                K_II = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VI
                K_IS = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VS
                K_SI = (self.spaces.T_SV @ self._system_blocks.K) @ self.spaces.T_VI
                K_SS = self.spaces.T_SV @ self._system_blocks.K @ self.spaces.T_VS
                K = _sparse.bmat(
                    [
                        [K_II, K_IS],
                        [K_SI, K_SS]
                    ],
                    'csc'
                )            

            # Eliminate interior DoFs
            _sparse.linalg.use_solver(useUmfpack=True)
            K_LU = _sparse.linalg.factorized(K)
        
        else:
            K_LU = lu_solve
        
        if self._formulation == 'teth':
            def matvec_fun(x):
                KBx = K_LU(_np.concatenate((_np.zeros(in_size), self._system_blocks.B @ x)))
                return (
                    0.5 * (self._system_blocks.P @ x + self._system_blocks.Q @ x)
                    - 0.5 * (
                        self._system_blocks.P @ KBx[in_size:]
                        - self._system_blocks.Q @ KBx[in_size:]
                    )
                )
        else:
            def matvec_fun(x):
                KBx = K_LU(_np.concatenate((_np.zeros(in_size), self._system_blocks.B @ x)))
                return self._system_blocks.Q @ x - self._system_blocks.P @ KBx[in_size:]

        system_operator = _sparse.linalg.LinearOperator(
            shape = 2 * (bi_size,),
            matvec = matvec_fun,
            dtype = _np.complex128
        )

        if counter is None:
            sol, info = solver(system_operator, self._rhs[fe_size:], tol=solver_tol)
        else:
            sol, info = solver(system_operator, self._rhs[fe_size:], callback=counter, tol=solver_tol)

        self.sol_E = -K_LU(_np.concatenate((_np.zeros(in_size), self._system_blocks.B @ sol)))
        self.sol_H = sol

        return info, lu_solve


class FEBISystemACA(FEBISystem):
    """
    FEBI system where the BI blocks are assembled using the adaptive cross approximation (ACA).
    """

    def __init__(self, frequency, computation_volume, formulation=None):

        if formulation is None:
            super().__init__(frequency, computation_volume)
        else:
            super().__init__(frequency, computation_volume, formulation)

        # Far interactions used in solution
        self._far_operator = None


    def assemble(
        self, recompress=True, target_points=None, max_points=None, max_level=16, tolerance=1e-3,
        quad_order_singular=5
    ):
        """
        Assemble system using a multilevel ACA for the BI blocks. An octree is generated for this
        with target_points DoFs per group at the lowest level. The octree is generated iteratively
        to arrive at a tree with a mean value for the DoFs per group close to this.

        Parameters
        ----------
        recompress : bool, optional
            Whether to recompress ACA blocks using QR and SVD, by default True.
        target_points : int, optional
            Target for the mean number of points per group, by default None. The default will set
            this to the square root of the number of BI DoFs.
        max_points : int, optional
            Maximum number of points per groups, by default None. This will override the iteration
            for obtaining mean number of points per groups.
        max_level : int, optional
            Maximum level for octree, by default 16.
        tolerance : float, optional
            Tolerance for the ACA and recompression SVD, by default 1e-3.
        quad_order_singular : int, optional
            Quadrature order to use in DEMCEM singular integral computations, by default 5.
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
            self._k0, self.spaces.bi_meshdata, self.spaces.bi_basisdata, recompress,
            target_points, max_points, max_level, tolerance, quad_order_singular
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
        self, solver=None, preconditioner=None, new_prec=False, right_prec=True, counter=None,
        solver_tol=1e-5, bi_scale=1.
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
            Preconditioner to use, with the default being fe2ms.preconditioners.direct.
            Preconditioners can be generated by functions in fe2ms.preconditioners, or created
            manually. A manually created preconditioner should be a function which takes a solution
            vector as its argument and returns the preconditioned vector.
        new_prec : bool, optional
            Whether to generate a new preconditioner even if there is an existing one already,
            by default False.
        right_prec : bool, optional
            Whether to use right preconditioning instead of left, by default True.
        counter : function, optional
            Callback function to scipy iterative solver, by default None. Note that different
            solvers call this differently. For more info, see scipy documentation.
        solver_tol : float, optional
            (Relative) tolerance of iterative solver, by default 1e-5.
        bi_scale : complex, optional
            Scaling factor for the BI equation, by default 1. Not applied to EJ formulation.

        Returns
        -------
        info : int
            0 if solver converged, >0 if solver did not converge,
            <0 if illegal input or breakdown of solver.
        """

        if self._rhs is None:
            raise Exception('No right-hand-side set!')

        if self._system_blocks is None:
            LOGGER.info('System not assembled, doing that before solving')
            self.assemble()
        
        if solver is None:
            solver = _sparse_linalg.lgmres

        # Generate or load preconditioner
        # FIXME: Change how preconditioners are used. They should probably not be loaded like this
        if preconditioner is None:
            if self.M_prec is None or new_prec:
                self.M_prec = _precs.direct(self)
        else:
            self.M_prec = preconditioner
        
        if self._formulation == 'ej':
            bi_scale = 1.

        # LinearOperator for matrix-vector multiplication such that sparse blocks are kept that way
        if self._formulation != 'vs-efie':
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
                        bi_scale * self._system_blocks.P @ x[in_size:-bi_size]
                        + bi_scale * self._far_operator.matvec_Kop(x[in_size:-bi_size])
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                        + bi_scale * self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        bi_scale * self._system_blocks.P @ x[in_size:-bi_size]
                        + bi_scale * self._far_operator.matvec_Kop(x[in_size:-bi_size])
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                        + bi_scale * self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
        elif self._formulation == 'vs-efie':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        bi_scale * self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + bi_scale * self._far_operator.matvec_Kop(self.spaces.T_SV @ x[:fe_size])
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                        + bi_scale * self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        self._system_blocks.K @ x[:fe_size] +
                        self.spaces.T_VS @ (self._system_blocks.B @ x[-bi_size:]),
                        bi_scale * self._system_blocks.P @ (self.spaces.T_SV @ x[:fe_size])
                        + bi_scale * self._far_operator.matvec_Kop(self.spaces.T_SV @ x[:fe_size])
                        + bi_scale * self._system_blocks.Q @ x[-bi_size:]
                        + bi_scale * self._far_operator.matvec_Lop(x[-bi_size:])
                    ))
        elif self._formulation == 'ej':
            eta0 = _np.sqrt(_mu0 / _eps0)
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
                        - 1j * self._k0 * eta0 * (
                            self._system_blocks.P.T @ x[-bi_size:]
                            + self._far_operator.matvec_Kop_T(x[-bi_size:])
                        ),
                        - 1j * self._k0 * eta0 * (
                            self._system_blocks.P @ x[in_size:-bi_size]
                            + self._far_operator.matvec_Kop(x[in_size:-bi_size])
                        )
                        - 1j * self._k0 * eta0**2 * (
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
                        - 1j * self._k0 * eta0 * (
                            self._system_blocks.P.T @ x[-bi_size:]
                            + self._far_operator.matvec_Kop_T(x[-bi_size:])
                        ),
                        - 1j * self._k0 * eta0 * (
                            self._system_blocks.P @ x[in_size:-bi_size]
                            + self._far_operator.matvec_Kop(x[in_size:-bi_size])
                        )
                        - 1j * self._k0 * eta0**2 * (
                            self._system_blocks.Q @ x[-bi_size:]
                            + self._far_operator.matvec_Lop(x[fe_size:])
                        )
                    ))
        elif self._formulation == 'teth':
            if right_prec:
                def matvec_fun(x):
                    x = self.M_prec(x)
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        bi_scale * 0.5 * (
                            self._system_blocks.P @ x[in_size:-bi_size]
                            - self._system_blocks.Q @ x[in_size:-bi_size]
                        )
                        + bi_scale * 0.5 * (
                            self._far_operator.matvec_Kop(x[in_size:-bi_size])
                            - self._far_operator.matvec_Lop(x[in_size:-bi_size])
                        )
                        + bi_scale * 0.5 * (
                            self._system_blocks.P @ x[-bi_size:]
                            + self._system_blocks.Q @ x[-bi_size:]
                        )
                        + bi_scale * 0.5 * (
                            self._far_operator.matvec_Kop(x[-bi_size:])
                            + self._far_operator.matvec_Lop(x[-bi_size:])
                        )
                    ))
            else:
                def matvec_fun(x):
                    return _np.concatenate((
                        K_II @ x[:in_size] + K_IS @ x[in_size:-bi_size],
                        K_SI @ x[:in_size] + K_SS @ x[in_size:-bi_size]
                        + self._system_blocks.B @ x[-bi_size:],
                        bi_scale * 0.5 * (
                            self._system_blocks.P @ x[in_size:-bi_size]
                            - self._system_blocks.Q @ x[in_size:-bi_size]
                        )
                        + bi_scale * 0.5 * (
                            self._far_operator.matvec_Kop(x[in_size:-bi_size])
                            - self._far_operator.matvec_Lop(x[in_size:-bi_size])
                        )
                        + bi_scale * 0.5 * (
                            self._system_blocks.P @ x[-bi_size:]
                            + self._system_blocks.Q @ x[-bi_size:]
                        )
                        + bi_scale * 0.5 * (
                            self._far_operator.matvec_Kop(x[-bi_size:])
                            + self._far_operator.matvec_Lop(x[-bi_size:])
                        )
                    ))

        system_operator = _sparse_linalg.LinearOperator(
            shape = 2 * (self.spaces.fe_size + self.spaces.bi_size,),
            matvec = matvec_fun,
            dtype = _np.complex128
        )

        if counter is None:
            if right_prec:
                sol, info = solver(system_operator, bi_scale * self._rhs, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(system_operator, bi_scale * self._rhs, M=M_op, tol=solver_tol)
        else:
            if right_prec:
                sol, info = solver(system_operator, bi_scale * self._rhs, callback=counter, tol=solver_tol)
                sol = self.M_prec(sol)
            else:
                M_op = _sparse_linalg.LinearOperator(
                    shape = 2 * (fe_size + bi_size,),
                    matvec = self.M_prec,
                    dtype = _np.complex128
                )
                sol, info = solver(
                    system_operator, bi_scale * self._rhs, M=M_op, callback=counter, tol=solver_tol
                )

        self.sol_E = sol[:self.spaces.fe_size]
        self.sol_H = sol[self.spaces.fe_size:]

        if self._formulation == 'ej':
            self.sol_H *= _np.sqrt(_mu0 / _eps0)

        return info
    

    def solve_semidirect(
        self, solver=None, counter=None, solver_tol=1e-5, lu_solve=None
    ):
        """
        Solve system by first eliminating the FE part using a sparse LU factorization, and then
        using this with the BI part in an iterative solver.

        No preconditioning options at this time. Not available for 'ej' formulation.

        This method is suitable for problems with large FE part and small BI part, or  particularly
        ill-conditioned problems with small enough BI part.
        Will assemble if not already done.

        Parameters
        ----------
        solver : function, optional
            Iterative solver to use, by default scipy.sparse.linalg.lgmres. This should have the
            same call structure as other iterative solvers in scipy.sparse.linalg.
        counter : function, optional
            Callback function to scipy iterative solver, by default None. Note that different
            solvers call this differently. For more info, see scipy documentation.
        solver_tol : float, optional
            (Relative) tolerance of iterative solver, by default 1e-5.
        lu_solve : function, optional
            Function corresponding to 'solve' operation for the LU factorization of the FE part.
            This is used instead of computing new factorizations using the system matrix.

        Returns
        -------
        info : int
            0 if solver converged, >0 if solver did not converge,
            <0 if illegal input or breakdown of solver.
        lu_solve : function
            Function corresponding to 'solve' operation for the LU factorization of the FE part
            which was used in the solution
        """

        if self._rhs is None:
            raise Exception('No right-hand-side set!')

        if self._system_blocks is None:
            LOGGER.info('System not assembled, doing that before solving')
            self.assemble()
        
        if solver is None:
            solver = _sparse_linalg.lgmres

        if self._formulation == 'ej':
            raise Exception('Semidirect solution not available for ej formulation')
        
        fe_size = self.spaces.fe_size
        bi_size = self.spaces.bi_size
        in_size = fe_size - bi_size
        
        if lu_solve is None:
            if self._formulation != 'vs-efie':
                K_II = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VI
                K_IS = self.spaces.T_IV @ self._system_blocks.K @ self.spaces.T_VS
                K_SI = (self.spaces.T_SV @ self._system_blocks.K) @ self.spaces.T_VI
                K_SS = self.spaces.T_SV @ self._system_blocks.K @ self.spaces.T_VS
                K = _sparse.bmat(
                    [
                        [K_II, K_IS],
                        [K_SI, K_SS]
                    ],
                    'csc'
                )            

            # Eliminate interior DoFs
            _sparse.linalg.use_solver(useUmfpack=True)
            K_LU = _sparse.linalg.factorized(K)
        
        else:
            K_LU = lu_solve
        
        if self._formulation == 'teth':
            def matvec_fun(x):
                KBx = K_LU(_np.concatenate((_np.zeros(in_size), self._system_blocks.B @ x)))
                return (
                    0.5 * (self._system_blocks.P @ x + self._system_blocks.Q @ x)
                    + 0.5 * (self._far_operator.matvec_Kop(x) + self._far_operator.matvec_Lop(x))
                    - 0.5 * (
                        self._system_blocks.P @ KBx[in_size:]
                        - self._system_blocks.Q @ KBx[in_size:]
                    )
                    - 0.5 * (
                        self._far_operator.matvec_Kop(KBx[in_size:])
                        - self._far_operator.matvec_Lop(KBx[in_size:])
                    )
                )
        else:
            def matvec_fun(x):
                KBx = K_LU(_np.concatenate((_np.zeros(in_size), self._system_blocks.B @ x)))
                return (
                    self._system_blocks.Q @ x + self._far_operator.matvec_Lop(x)
                    - self._system_blocks.P @ KBx[in_size:]
                    - self._far_operator.matvec_Kop(KBx[in_size:])
                )

        system_operator = _sparse.linalg.LinearOperator(
            shape = 2 * (bi_size,),
            matvec = matvec_fun,
            dtype = _np.complex128
        )

        if counter is None:
            sol, info = solver(system_operator, self._rhs[fe_size:], tol=solver_tol)
        else:
            sol, info = solver(system_operator, self._rhs[fe_size:], callback=counter, tol=solver_tol)

        self.sol_E = -K_LU(_np.concatenate((_np.zeros(in_size), self._system_blocks.B @ sol)))
        self.sol_H = sol

        return info, lu_solve
