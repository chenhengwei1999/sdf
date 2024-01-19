import meshio

# stl_mesh = meshio.read('out.stl')
stl_mesh = meshio.read('out.stl')

print(stl_mesh.points, type(stl_mesh.points))
print(stl_mesh.points.shape)