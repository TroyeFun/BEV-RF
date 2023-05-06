import open3d as o3d
import numpy as np


def get_o3d_mesh(tsdf_vol, triangle_preserve_ratio=1.0, with_color=True):
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertex_normals = o3d.utility.Vector3dVector(norms)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if with_color:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    else:
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.compute_vertex_normals()
    mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * triangle_preserve_ratio))
    return mesh


def visualize_mesh(tsdf_vol, triangle_preserve_ratio=1.0, with_color=True):
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh = get_o3d_mesh(tsdf_vol, triangle_preserve_ratio, with_color)
    o3d.visualization.draw_geometries([mesh, origin_frame])


def get_o3d_point_cloud(tsdf_vol):
    verts, colors = tsdf_vol.get_point_cloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255)
    return pcd


def visualize_point_cloud(tsdf_vol):
    pcd = get_o3d_point_cloud(tsdf_vol)
    o3d.visualization.draw_geometries([pcd])


def visualize_voxel(tsdf_vol, from_point_cloud=True):
    if from_point_cloud:
        pcd = get_o3d_point_cloud(tsdf_vol)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.15)
    else:
        mesh = get_o3d_mesh(tsdf_vol)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])


def image2pcd(color, depth, intrinsics):
    n_rows, n_cols = color.shape[:2]
    xs, ys = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    pixels = np.stack([xs, ys], axis=2).reshape(-1, 2)
    depth = depth.reshape(-1, 1)
    inv_K = np.linalg.inv(intrinsics)
    points = depth * np.dot(inv_K[:3, :3], np.concatenate([pixels, np.ones((pixels.shape[0], 1))], axis=1).T).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).astype(np.float64) / 255)
    return pcd


def smooth_mesh(mesh, n_iter=10, method='laplacian'):
    assert method in ['simple', 'laplacian', 'taubin']
    funcs = {
        'simple': o3d.geometry.TriangleMesh.filter_smooth_simple,
        'laplacian': o3d.geometry.TriangleMesh.filter_smooth_laplacian,
        'taubin': o3d.geometry.TriangleMesh.filter_smooth_taubin,
    }
    mesh.compute_vertex_normals()
    mesh = funcs[method](mesh, number_of_iterations=n_iter)
    return mesh
