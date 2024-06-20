"""
Script which is used to create mesh in rotor.msh for a wind turbine rotor using STEP files.
"""
import numpy as np
import gmsh
from scipy.constants import speed_of_light as c0

import fe2ms

f0 = 30e6
lam = c0 / f0

# Define tags which are stored in the mesh file (and need to be entered in the FE-BI code)
tag_air = 1
tag_glass = 2
tag_ext = 3

# Set mesh sizes for different regions (0.05 * wavelength in medium)
mesh_frac = 0.05
mesh_sizes = [
    lam * mesh_frac,
    lam * mesh_frac / np.sqrt( fe2ms.materials.FR4_1MHz[0].real)
]

####################################################################################################
# Load STEP files in Gmsh and create geometry with tags based on the two input files (air and glass)

gmsh.initialize()
gmsh.model.add('rotor')

# Import STEP geometry for both materials with the unit meter used when interpreting STEP files
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')
air = gmsh.model.occ.import_shapes('rotor_air.step')
glass = gmsh.model.occ.import_shapes('rotor_glass.step')

# Make sure that all boundaries are conformal
gmsh.model.occ.fragment(air, glass)
gmsh.model.occ.synchronize()

# Remove lower dimensional objects which can exist in STEP files but are not used here
gmsh.model.removeEntities(gmsh.model.getEntities(2), True)
gmsh.model.removeEntities(gmsh.model.getEntities(1), True)
gmsh.model.removeEntities(gmsh.model.getEntities(0), True)

# Find boundaries of all constituent parts and the external boundary
glass_boundary = set(gmsh.model.getBoundary(glass, oriented=False))
air_boundary = set(gmsh.model.getBoundary(air, oriented=False))
ext_boundary = glass_boundary.difference(air_boundary)

# Add physical groups corresponding to material volumes and external boundary
gmsh.model.addPhysicalGroup(3, [v[1] for v in air], tag_air)
gmsh.model.addPhysicalGroup(3, [v[1] for v in glass], tag_glass)
gmsh.model.addPhysicalGroup(2, [b[1] for b in ext_boundary], tag_ext)

####################################################################################################
# Create mesh for the geometry in gmsh

# Iterate over coating volumes and their mesh sizes to set mesh size fields for the two regions
# Fields for different materials
regions = [air, glass]
fields = []
for reg, size in zip(regions, mesh_sizes):

    # Set mesh size field for current material
    f_size = gmsh.model.mesh.field.add('MathEval')
    gmsh.model.mesh.field.setString(f_size, 'F', str(size))

    # Restrict that field to the current coating (volume, surfaces and curves)
    # This will ensure that the current size is only applied at the current coating
    f_restr = gmsh.model.mesh.field.add('Restrict')
    gmsh.model.mesh.field.setNumber(f_restr, 'InField', f_size)
    gmsh.model.mesh.field.setNumbers(f_restr, 'VolumesList', [v[1] for v in reg])
    for v in reg:
        surfs = gmsh.model.getAdjacencies(*v)[1].tolist()
        gmsh.model.mesh.field.setNumbers(f_restr, 'SurfacesList', surfs)
        curves = []
        for s in surfs:
            curves += gmsh.model.getAdjacencies(2, s)[1].tolist()
        gmsh.model.mesh.field.setNumbers(f_restr, 'CurvesList', curves)
    fields.append(f_restr)

# Set background field  (the field which is used to generate the mesh) to be minimum of fields
# present at each point
f_min = gmsh.model.mesh.field.add('Min')
gmsh.model.mesh.field.setNumbers(f_min, 'FieldsList', fields)
gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

# Disregard mesh settings not needed for this approach
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Save mesh file (uncomment commented line to also save .vtk file which can be opened in Paraview)
gmsh.write('rotor.msh')
# gmsh.write('rotor.vtk')

gmsh.finalize()
