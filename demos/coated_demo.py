"""
Demo script for fe2ms FE-BI code applied to scattering by a PEC sphere with a two-layer coating.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

import fe2ms

f0 = 1e9

# Tag different regions and set material parameters. These need to correspond to the tags in the
# mesh file.
materials = {1: fe2ms.materials.VACUUM, 2: fe2ms.materials.PLA}
pec_surface = [3]
ext_surface = [4]

####################################################################################################
# Create FE-BI system and perform assembly

# ComputationalVolume data structure contains all data about the mesh and its regions
# This uses the mesh file coated.msh which is created using gmsh in create_mesh_coated.py
cv = fe2ms.ComputationVolume('coated.msh', materials, ext_surface, pec_surface)

# System which will use the ACA for acceleration and a formulation based on the EFIE with volume
# degrees of freedom split into exclusively in interior and exclusively on boundary ('is-efie')
system = fe2ms.FEBISystemACA(f0, cv, 'is-efie')

# Commented line below is for a full (unaccelerated) system with the same formulation as above
# system = fe2ms.FEBISystemFull(f0, cv, 'is-efie')

print('Start connection and assembly')
tick = time.perf_counter()

# Assemble system. A connection between FE and BI degrees of freedom is first done, then assembly is
# performed using the selected method (full or ACA)
system.assemble()

print(f'Connection and assembly time: {time.perf_counter()-tick:.1f} s')
print(f'FE size: {system.spaces.fe_size}, BI size: {system.spaces.bi_size}')

# Set incident wave to be a plane wave with
# - Amplitude 1 V/m, 
# - Polarization along z
# - Propagation along x
system.set_source_plane_wave(1, np.array([0, 0, 1]), np.array([1, 0, 0]))

####################################################################################################
# Solve system and compute bistatic RCS

print('Start solution')
tick = time.perf_counter()

# Solve using default settings
system.solve()

print(f'Solution time: {time.perf_counter()-tick:.1f} s')

# Angles for E-plane (xz) and H-plane (xy)
theta_e = np.linspace(-np.pi, np.pi, 400)
phi_e = np.full_like(theta_e, 0)
phi_h = np.linspace(0, 2*np.pi, 400)
theta_h = np.full_like(phi_h, np.pi/2)

print('Start RCS computation')
tick = time.perf_counter()

# Compute bistatic RCS in E-plane and H-plane
rcs_e = system.compute_rcs(theta_e, phi_e)
rcs_h = system.compute_rcs(theta_h, phi_h)

print(f'RCS computation time: {time.perf_counter()-tick:.1f} s')

####################################################################################################
# Plot RCS and save figures

plt.figure()
plt.title('E-plane')
plt.plot(np.degrees(theta_e), 10*np.log10(rcs_e))
plt.xlim((-180, 180))
plt.ylabel('$\\sigma$ (dBsm)')
plt.xlabel('$\\theta$ (deg)')
plt.savefig('rcs_eplane.png')

plt.figure()
plt.title('H-plane')
plt.plot(np.degrees(phi_h), 10*np.log10(rcs_h))
plt.xlim((0, 360))
plt.ylabel('$\\sigma$ (dBsm)')
plt.xlabel('$\\phi$ (deg)')
plt.savefig('rcs_hplane.png')
