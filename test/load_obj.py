from pytorch3d.io import load_obj

import torch

import trimesh

f = load_obj("/home/chw/Documents/sdf/out.obj")

verts, faces_all, file_property = load_obj("/home/chw/Documents/sdf/out.obj")

faces = faces_all[0]

# visualize the mesh
plotly_mesh = trimesh.Trimesh(
    verts.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
    vertex_colors=[100, 100, 255],
)

plotly_mesh.show()

def rotate_points(points, angle):
    # Rotate 3D points around the z-axis
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=points.dtype, device=points.device)

    rotated_points = torch.matmul(points, rotation_matrix)
    return rotated_points


