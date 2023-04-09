import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
import torch


def show_img(img):
    """Show image using matplotlib.
    Args:
        img (np.ndarray | PIL.Image | str): Image to show.
    """
    if isinstance(img, str):
        img = Image.open(img)
    plt.imshow(img)
    plt.show()


def dump_tensor(save_name, tensor):
    """Dump tensor to numpy array.
    Args:
        tensor (torch.Tensor): Tensor to dump.
    Returns:
        np.ndarray: Numpy array of the tensor.
    """
    os.makedirs('tmp_data', exist_ok=True)
    array = tensor.detach().cpu().numpy()
    np.save(os.path.join('tmp_data', save_name + '.npy'), array)


def get_pcd(points, colors=None):
    """Get open3d point cloud from points and colors.
    Args:
        points (np.ndarray): Points of point cloud.
        colors (np.ndarray, optional): Colors of point cloud.
            Defaults to None.
    Returns:
        open3d.geometry.PointCloud: Open3d point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def show_pcd(pcd):
    """Show open3d point cloud.
    Args:
        pcd (open3d.geometry.PointCloud): Open3d point cloud.
    """
    o3d.visualization.draw_geometries([pcd])


def show_pcds(pcds):
    """Show multiple open3d point clouds.
    Args:
        pcds (list[open3d.geometry.PointCloud]): Open3d point clouds.
    """
    o3d.visualization.draw_geometries(pcds)


def save_mock_voxel_feature(voxel_feature, cam_pts, scene_range, sample_feats_3d_fn):
    C, D, H, W = voxel_feature.shape
    mock_voxel_feature = torch.arange(3 * D * H * W).type_as(voxel_feature).reshape(3, D, H, W)
    mock_voxel_feature /= (3 * D * H * W)
    mock_pts_feature = sample_feats_3d_fn(mock_voxel_feature, cam_pts, scene_range)
    dump_tensor('mock_voxel_feature', mock_voxel_feature)
    dump_tensor('mock_pts_feature', mock_pts_feature)
    dump_tensor('cam_pts', cam_pts)