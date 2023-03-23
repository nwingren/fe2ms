"""
Test script for DEMCEM bindings. ./test/demcem_example_runner.cpp needs to be built using CMake and
run to generate the reference results necessary for this to run.

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

import numpy as np
import demcem_bindings

def test_ss_ea_nxrwg():

    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.0, 0.1, 0.0])
    r3 = np.array([0.0, 0.0, 0.1])
    r4 = np.array([0.1, 0.0, 0.0])
    k0 = 2 * np.pi
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ss_ea_nxrwg(r1, r2, r3, r4, k0, 5, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ss_ea_nxrwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)


def test_ss_ea_rwg():

    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.0, 0.1, 0.0])
    r3 = np.array([0.0, 0.0, 0.1])
    r4 = np.array([0.1, 0.0, 0.0])
    k0 = 2 * np.pi
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ss_ea_rwg(r1, r2, r3, r4, k0, 5, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ss_ea_rwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)


def test_ss_va_nxrwg():

    r1 = np.array([0.1, 0.1, 0.1])
    r2 = np.array([0.2, 0.1, 0.1])
    r3 = np.array([0.1, 0.2, 0.1])
    r4 = np.array([0.0, 0.1, 0.2])
    r5 = np.array([0.0, 0.2, 0.2])
    k0 = 2 * np.pi
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ss_va_nxrwg(r1, r2, r3, r4, r5, k0, 5, 5, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ss_va_nxrwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)


def test_ss_va_rwg():

    r1 = np.array([0.1, 0.1, 0.1])
    r2 = np.array([0.2, 0.1, 0.1])
    r3 = np.array([0.1, 0.2, 0.1])
    r4 = np.array([0.0, 0.1, 0.2])
    r5 = np.array([0.0, 0.2, 0.2])
    k0 = 2 * np.pi
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ss_va_rwg(r1, r2, r3, r4, r5, k0, 5, 5, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ss_va_rwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)


def test_ws_ea_rwg():

    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.5, 0.0, 0.0])
    r3 = np.array([0.0, 0.0, 0.5])
    r4 = np.array([0.0, 0.5, 0.0])
    k0 = 1.0
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ws_ea_rwg(r1, r2, r3, r4, k0, 5, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ws_ea_rwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)


def test_ws_st_rwg():
    
    r1 = np.array([0.0, 0.0, 0.5])
    r2 = np.array([0.0, 0.0, 0.0])
    r3 = np.array([0.5, 0.0, 0.0])
    k0 = 2 * np.pi
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ws_st_rwg(r1, r2, r3, k0, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ws_st_rwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)


def test_ws_va_rwg():

    r1 = np.array([1.0, 1.0, 1.0])
    r2 = np.array([2.0, 1.0, 1.0])
    r3 = np.array([1.0, 2.0, 1.0])
    r4 = np.array([0.0, 1.0, 1.0])
    r5 = np.array([0.0, 2.0, 1.0])
    k0 = 2 * np.pi
    result = np.zeros(9, dtype=np.complex128)

    demcem_bindings.ws_va_rwg(r1, r2, r3, r4, r5, k0, 5, 5, 5, result)

    refvals = np.loadtxt('test/build/demcem_ref_ws_va_rwg.txt')
    refvals = refvals[:,0] + 1j * refvals[:,1]

    assert np.allclose(result, refvals)
