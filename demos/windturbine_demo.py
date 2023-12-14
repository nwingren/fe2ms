"""
Demo script for fe2ms FE-BI code applied to scattering by a wind turbine rotor.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import fe2ms

f0 = 30e6

mats = {1: fe2ms.materials.VACUUM, 2: fe2ms.materials.FR4_1MHz}
ext = [3]

####################################################################################################
# Create FE-BI system and perform assembly

cv = fe2ms.ComputationVolume('rotor.msh', mats, ext)
system = fe2ms.FEBISystemACA(f0, cv)

print('Start connection and assembly')
tick = time.perf_counter()
system.assemble()
print(f'Connection and assembly time: {time.perf_counter()-tick:.1f} s')
print(f'FE size: {system.spaces.fe_size}, BI size: {system.spaces.bi_size}')

####################################################################################################
# Solve for plane waves and compute monostatic RCS

# Create preconditioner ahead of solution since it can be reused for all right-hand-sides
tick = time.perf_counter()
M_prec = fe2ms.preconditioners.direct(system)
print(f'Preconditioner generation time: {time.perf_counter()-tick:0.1f} s')

# Callback function for counting iterations in iterative solver
iters = 0
def it_counter(x):
    global iters
    iters += 1

# Varying phi angle (50 angles from -pi/6 to pi/2)
# This is equivalent to fix incidence and rotating rotor
print('Compute monostatic RCS')
phi = np.linspace(-np.pi/6, np.pi/2, 50)
rcs = np.empty_like(phi)
tick_outer = time.perf_counter()

# Iterate over phi angles and compute monostatic RCS for each incidence
for i, p in enumerate(phi):

    # xy-polarized plane wave with phi=p incidence
    system.set_source_plane_wave(
        1., np.array([-np.sin(p), np.cos(p), 0]), np.array([np.cos(p), np.sin(p), 0])
    )

    # Solve system for that incidence and compute the monostatic RCS
    iters = 0
    tick = time.perf_counter()
    system.solve_iterative(preconditioner=M_prec, counter=it_counter)
    rcs[i] = system.compute_rcs(np.array([np.pi/2]), np.array([p + np.pi]))
    print(f'RCS {i} of {phi.size}: {time.perf_counter()-tick:0.1f} s, {iters} iterations')

print(f'Total time: {time.perf_counter()-tick_outer:0.1f} s')

####################################################################################################
# Plot RCS

plt.figure()
plt.plot(np.degrees(phi), 10*np.log10(rcs))
plt.xlim((-30, 90))
plt.xticks(np.arange(-30,91,30))
plt.ylabel('$\\sigma$ (dBsm)')
plt.xlabel('$\\phi$ (deg)')
plt.title('Monostatic RCS, xy incidence and polarization')
plt.show()
