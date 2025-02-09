from multiprocessing import Pool
import os
import os.path as osp
import time

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

import test_utils
import test_utils.mesh_utils as mesh_utils
from tools.data_converter.generate_novel_depth import sample_novel_cam_poses, NOVEL_CAM_K
from tools.data_converter.tsdf_fusion import TSDFVolume

SCENE_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 10.0]
# SCENE_RANGE = [-51.2, -102.4, -5.0, 51.2, 102.4, 15.0]
VOXEL_SIZE = 0.2

mesh_dir, smooth_mesh_dir, tsdf_dir, color_dir, depth_dir, valid_ray_dir, vol_bounds, cam_poses = [None] * 8


def create_tsdf_dirs(exp_root):
    mesh_dir = osp.join(exp_root, 'novel_views', 'mesh')
    smooth_mesh_dir = osp.join(exp_root, 'novel_views', 'smooth_mesh')
    tsdf_dir = osp.join(exp_root, 'novel_views', 'tsdf')
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(smooth_mesh_dir, exist_ok=True)
    os.makedirs(tsdf_dir, exist_ok=True)
    return mesh_dir, smooth_mesh_dir, tsdf_dir


def single_depth2tsdf(i):
    mesh_file = osp.join(mesh_dir, f'{i:05d}.ply')
    smooth_mesh_file = osp.join(smooth_mesh_dir, f'{i:05d}.ply')
    tsdf_file = osp.join(tsdf_dir, f'{i:05d}.npy')
    tsdf_vol = TSDFVolume(vol_bounds, voxel_size=VOXEL_SIZE)
    for cid, pose in enumerate(cam_poses):
        # if cid not in [1, 4]:
        #     continue
        color_file = osp.join(color_dir, f'{i:05d}_{cid:02d}.png')
        depth_file = osp.join(depth_dir, f'{i:05d}_{cid:02d}.npy')
        valid_ray_file = osp.join(valid_ray_dir, f'{i:05d}_{cid:02d}.npy')
        depth = np.load(depth_file)
        color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
        valid_ray_ratio = np.load(valid_ray_file)
        mask = (valid_ray_ratio > 0.99)
        color *= mask[:, :, None]
        depth *= mask
        tsdf_vol.integrate(color, depth, NOVEL_CAM_K, pose, obs_weight=1.0)

    tsdf_vol.save_mesh(mesh_file)
    tsdf_grid, _ = tsdf_vol.get_volume()
    np.save(tsdf_file, tsdf_grid)
    mesh = mesh_utils.get_o3d_mesh(tsdf_vol, triangle_preserve_ratio=1.0)
    mesh_smooth = mesh_utils.smooth_mesh(mesh, n_iter=10, method='laplacian')
    o3d.io.write_triangle_mesh(smooth_mesh_file, mesh_smooth)

    # mesh_utils.visualize_mesh(tsdf_vol, triangle_preserve_ratio=0.04, with_color=True)
    # import ipdb; ipdb.set_trace()
    # pcd = mesh_utils.image2pcd(color, depth, NOVEL_CAM_K)
    # test_utils.show_pcd(pcd)


def depth2tsdf(exp_root):
    global color_dir, depth_dir, valid_ray_dir, mesh_dir, smooth_mesh_dir, tsdf_dir, vol_bounds, cam_poses
    (_, color_dir, depth_dir, valid_ray_dir
    ) = test_utils.create_novel_views_dir(osp.join(exp_root, 'novel_views'), create=False)
    mesh_dir, smooth_mesh_dir, tsdf_dir = create_tsdf_dirs(exp_root)

    cam_poses = sample_novel_cam_poses()
    vol_bounds = np.array(SCENE_RANGE).reshape(2, 3).T
    dataset_size = 1000
    res = list(tqdm(map(single_depth2tsdf, range(dataset_size)), total=dataset_size))
        

if __name__ == '__main__':
    exp_root = 'runs/230504-nerffusion-depth_reg0.01-lr0.0002-ray_pts64'
    depth2tsdf(exp_root)