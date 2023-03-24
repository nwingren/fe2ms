"""
Various preconditioners used for iterative solution of full FE-BI system.

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

import numpy as _np
from scipy import sparse as _sparse
from scipy.sparse import linalg as _sparse_linalg

from fe2ms.systems import FEBISystem as _FEBISystem


def direct(
    system: _FEBISystem, scale=None
):
    """
    Sparse approximation direct preconditioner.

    Uses sparsified BI blocks where only interactions within single triangles are included.
    A sparse approximation to the system matrix is constructed from these as well as the already
    sparse FE blocks. The approximation to the inverse is formed by computing the sparse LU
    decomposition of this sparse approximation, and applying it in matrix-vector-products.

    The direct LU decomposition uses the UMFPACK library with a sparse matrix,
    which gives good performance. Nevertheless, too large systems are ill-suited for direct solvers.
    In those cases, an iterative solver may be considered when computing the preconditioner.
    See sparse_approx_iterative_ilu.

    Parameters
    ----------
    system : FEBISystem
        System to precondition.
    scale : complex, optional
        Factor to scale sparisifed blocks by, by default None.
        If None, EFIE formulations are scaled by -2jk0 while EJ formulation is unscaled.

    Returns
    -------
    M : function
        Solve function which applies the preconditioner to a vector.
    """

    sparse_mat = _make_sparse_mat(
        system._formulation, system._k0, system._system_blocks, system.spaces,
        system.K_prec, system.L_prec, scale
    )

    _sparse_linalg.use_solver(useUmfpack=True)
    M_prec = _sparse_linalg.factorized(sparse_mat)
    del sparse_mat

    return M_prec


def iterative_ilu(
    system: _FEBISystem, scale=None,
    solver=None, solver_tol=1e-3,
    spilu_droprule='interp', spilu_droptol=1e-3, spilu_options={}
):
    """
    Sparse approximation preconditioner with iterative computation and inner ILU preconditioning.

    Uses sparsified BI blocks where only interactions within single triangles are included.
    A sparse approximation to the system matrix is constructed from these as well as the already
    sparse FE blocks. The approximation to the inverse is applied through iterative solution each
    time the preconditioner is applied to a vector. This inner iterative solver is in turn
    right-preconditioned by an ILU of the sparsified system.

    Note that this preconditioner is only intended for very large systems. For many cases, direct()
    has performed much better.

    Parameters
    ----------
    
    system : FEBISystem
        System to precondition.
    scale : complex, optional
        Factor to scale sparisifed blocks by, by default None.
        If None, EFIE formulations are scaled by -2jk0 while EJ formulation is unscaled.
    solver : function, optional
        Iterative solver to use, by default scipy.sparse.linalg.bicgstab. This should have the
        same call structure as other iterative solvers in scipy.sparse.linalg.
    solver_tol : float, optional
        (Relative) tolerance of iterative solver, by default 1e-3.
    spilu_droprule : str, optional
        Drop rule option in SuperLU SPILU, by default 'interp'.
    spilu_droptol : float, optional
        Drop tolerance in SuperLU SPILU, by default 1e-3.
    spilu_options : dict, optional
        Dictionary of additional SuperLU SPILU options, by default empty dict.

    Returns
    -------
    M : LinearOperator
        Linear operator acting as an approximaiton to the invese if applied
        in a matrix-vector-product.
    """

    if solver is None:
        solver = _sparse_linalg.bicgstab

    sparse_mat = _make_sparse_mat(
        system._formulation, system._k0, system._system_blocks, system.spaces,
        system.K_prec, system.L_prec, scale
    )

    M_prec_inner = _sparse_linalg.spilu(
        sparse_mat, drop_rule=spilu_droprule, drop_tol=spilu_droptol, options=spilu_options
    )
    precd_op = _sparse_linalg.LinearOperator(
        shape = 2 * (system.spaces.fe_size + system.spaces.bi_size,),
        matvec = lambda x: sparse_mat @ M_prec_inner.solve(x),
        dtype = _np.complex128
    )

    def precfun(x):
        psol, pinfo = solver(precd_op, x, tol=solver_tol, atol=0)
        if pinfo != 0:
            raise RuntimeError(
                'Inner loop (preconditioning) did not converge'
            )
        return M_prec_inner.solve(psol)

    return precfun


def _make_sparse_mat(formulation, k0, system_blocks, spaces, K_prec, L_prec, scale):
    """
    Makes sparsified matrix common to both direct and iterative preconditioner.
    """

    if scale is None:
        if formulation == 'ej':
            scale = 1
        else:
            scale = -2j * k0

    if formulation in ('is-efie', 'ej'):
        K_II = spaces.T_IV @ system_blocks.K @ spaces.T_VI
        K_IS = spaces.T_IV @ system_blocks.K @ spaces.T_VS
        K_SI = (spaces.T_SV @ system_blocks.K) @ spaces.T_VI
        K_SS = spaces.T_SV @ system_blocks.K @ spaces.T_VS

    if formulation == 'is-efie':
        sparse_mat = _sparse.bmat(
            [
                [K_II, K_IS, None],
                [K_SI, K_SS, system_blocks.B],
                [None, scale * K_prec, scale * L_prec]
            ],
            'csc'
        )
    elif formulation == 'vs-efie':
        sparse_mat = _sparse.bmat(
            [
                [system_blocks.K, spaces.T_IV @ system_blocks.B],
                [scale * K_prec, scale * L_prec]
            ],
            'csc'
        )
    elif formulation == 'ej':
        sparse_mat = _sparse.bmat(
            [
                [K_II, K_IS, None],
                [K_SI, K_SS + 1j * k0 * scale * L_prec, -1j * k0 * scale * K_prec],
                [None, -1j * k0 * scale * K_prec.T, -1j * k0 * scale * L_prec]
            ],
            'csc'
        )
    
    return sparse_mat
