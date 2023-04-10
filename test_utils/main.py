import ipdb
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA
import torch

import test_utils


def show_cam_pts_and_voxel_feature():
    cam_pts = np.load('tmp_data/cam_pts.npy')
    voxel_feature = np.load('tmp_data/mock_voxel_feature.npy')
    pts_feature = np.load('tmp_data/mock_pts_feature.npy')  # (n_pts, C)

    cam_pts = cam_pts.reshape(-1, 3)  # (n_pts, 3) -- (x, y, z)
    C, D, H, W = voxel_feature.shape
    voxel_feature = np.transpose(voxel_feature, (2, 3, 1, 0))  # (H, W, D, C) -- (x, y, z, feature)
    voxel_feature = voxel_feature.reshape(-1, C)  # (n_voxel, C)

    # pca = PCA(3)
    # pca.fit(voxel_feature)
    # voxel_color = pca.fit_transform(voxel_feature)
    # pts_color = pca.fit_transform(pts_feature)
    # min_value, max_value = voxel_color.min(), voxel_color.max()
    # voxel_color = (voxel_color - min_value) / (max_value - min_value)
    # pts_color = (pts_color - min_value) / (max_value - min_value)
    cmap = cm.get_cmap('Spectral')
    voxel_color = cmap(voxel_feature[:, 0] * 3)[:, :3]
    pts_color = cmap(pts_feature[:, 0] * 3)[:, :3]

    scene_range = ((-25.6, 25.6), (-51.2, 51.2), (-5.0, 3.0))
    axis_x = np.linspace(scene_range[0][0], scene_range[0][1], H + 1)[:-1] + 0.5 * (scene_range[0][1] - scene_range[0][0]) / H
    axis_y = np.linspace(scene_range[1][0], scene_range[1][1], W + 1)[:-1] + 0.5 * (scene_range[1][1] - scene_range[1][0]) / W
    axis_z = np.linspace(scene_range[2][0], scene_range[2][1], D + 1)[:-1] + 0.5 * (scene_range[2][1] - scene_range[2][0]) / D
    grid_x, grid_y, grid_z = np.meshgrid(axis_x, axis_y, axis_z, indexing='ij')
    voxel_grid = np.stack([grid_x, grid_y, grid_z], axis=-1)
    voxel_grid = voxel_grid.reshape(-1, 3)
    voxel_pcd = test_utils.get_pcd(voxel_grid, voxel_color)
    cam_pcd = test_utils.get_pcd(cam_pts, pts_color)
    ipdb.set_trace()
    test_utils.show_pcds([voxel_pcd, cam_pcd])
    test_utils.show_pcd(cam_pcd)
    test_utils.show_pcd(voxel_pcd)


def modify_ckpt_vtransform_nx():
    ckpt_path = 'pretrained/bevfusion-seg.pth'
    new_ckpt_path = 'pretrained/bevfusion-seg-half_x_range.pth'
    ckpt = torch.load(ckpt_path)
    ipdb.set_trace()
    ckpt['state_dict']['encoders.camera.vtransform.nx'][0] = 128
    torch.save(ckpt, new_ckpt_path)


if __name__ == '__main__':
    # show_cam_pts_and_voxel_feature()
    modify_ckpt_vtransform_nx()