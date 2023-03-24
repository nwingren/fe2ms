"""
Script which is used to create mesh in coated.msh for a PEC sphere with a two-layer coating.
"""
import numpy as np
import gmsh
from scipy.constants import speed_of_light as c0

import fe2ms

f0 = 1e9
lam = c0 / f0
radius_pec = 0.4 * lam
thickness_coat1 = 0.1 * radius_pec
thickness_coat2 = 0.2 * radius_pec

# Define tags which are stored in the mesh file (and need to be entered in the FE-BI code)
tag_coat1 = 1
tag_coat2 = 2
tag_pec = 3
tag_ext = 4

# Set mesh sizes for different regions
mesh_sizes = [
    lam / 15,
    lam / 15 / np.sqrt(fe2ms.materials.PLA[0].real)
]

####################################################################################################
# Create geometry in gmsh

gmsh.initialize()
gmsh.model.add('coated')

# Create PEC sphere
sphere_pec = gmsh.model.occ.addSphere(0, 0, 0, radius_pec)

# Create coating 1 as difference between an outer sphere and the PEC sphere
sphere_coat1 = gmsh.model.occ.addSphere(0, 0, 0, radius_pec + thickness_coat1)
vol_coat1 = gmsh.model.occ.cut([(3, sphere_coat1)], [(3, sphere_pec)])[0]

# Create coating 2 as difference between an outer sphere and a dummy sphere
sphere_coat2 = gmsh.model.occ.addSphere(0, 0, 0, radius_pec + thickness_coat1 + thickness_coat2)
sphere_dummy = gmsh.model.occ.addSphere(0, 0, 0, radius_pec + thickness_coat1)
vol_coat2 = gmsh.model.occ.cut([(3, sphere_coat2)], [(3, sphere_dummy)])[0]

# Ensure that boundaries between the two coating volumes are conformal
gmsh.model.occ.fragment(vol_coat1, vol_coat2)
gmsh.model.occ.synchronize()

# Find all boundaries of the coatings and deduce those corresponding to PEC and external
boundaries1 = set(gmsh.model.getBoundary(vol_coat1, oriented=False))
boundaries2 = set(gmsh.model.getBoundary(vol_coat2, oriented=False))
boundary_pec = boundaries1.difference(boundaries2)
boundary_ext = boundaries2.difference(boundaries1)

# Add physical groups corresponding to material volumes, external and PEC boundaries
gmsh.model.addPhysicalGroup(3, [vol_coat1[0][1]], tag_coat1)
gmsh.model.addPhysicalGroup(3, [vol_coat2[0][1]], tag_coat2)
gmsh.model.addPhysicalGroup(2, [b[1] for b in boundary_pec], tag_pec)
gmsh.model.addPhysicalGroup(2, [b[1] for b in boundary_ext], tag_ext)

####################################################################################################
# Create mesh for the geometry in gmsh

# Iterate over coating volumes and their mesh sizes to set mesh size fields for the two regions
fields = []
for v, size in zip(vol_coat1 + vol_coat2, mesh_sizes):

    # Set mesh size field for current material
    f_size = gmsh.model.mesh.field.add('MathEval')
    gmsh.model.mesh.field.setString(f_size, 'F', str(size))

    # Restrict that field to the current coating (volume, surfaces and curves)
    # This will ensure that the current size is only applied at the current coating
    f_restr = gmsh.model.mesh.field.add('Restrict')
    gmsh.model.mesh.field.setNumber(f_restr, 'InField', f_size)
    gmsh.model.mesh.field.setNumbers(f_restr, 'VolumesList', [v[1]])
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

# Disregard meth settings not needed for this approach
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Save mesh file (uncomment commented line to also save .vtk file which can be opened in Paraview)
gmsh.write('coated.msh')
# gmsh.write('coated.vtk')

gmsh.finalize()