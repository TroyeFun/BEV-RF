import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import mmcv
from mmcv.runner import get_dist_info
import numpy as np
import open3d as o3d
from open3d.geometry import TriangleMesh as T
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv
import torch

from test_utils import visualize_novel_views, create_novel_views_dir

# NOVEL_IMG_SIZE = (800, 300)
NOVEL_IMG_SIZE = (800, 300)
FOV_H, FOV_V = np.pi * 60 / 180, np.pi * 20 / 180  # 120 degree horizontal FOV, 60 degree vertical FOV
NOVEL_CAM_K = np.array([
    [NOVEL_IMG_SIZE[0] / 2 / np.sqrt(np.tan(FOV_H / 2)), 0, NOVEL_IMG_SIZE[0] / 2],
    [0, NOVEL_IMG_SIZE[1] / 2 / np.sqrt(np.tan(FOV_V / 2)), NOVEL_IMG_SIZE[1] / 2],
    [0, 0, 1]])  


def sample_novel_cam_poses():
    lidar2front_cam = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    lidar2back_cam = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    
    forward_trans = 3.0
    rightward_trans = [-3.0, 0, 3.0]
    front_rads = [-np.pi / 6, 0, np.pi / 6]
    back_rads = [np.pi / 6, 0, -np.pi / 6]

    def _generate_cam_poses(forward_trans, lidar2cam, rads):
        T_base = np.eye(4)
        T_base[1, 3] = forward_trans
        poses = []
        for i in range(3):
            T_rightward = np.eye(4)
            T_rightward[0, 3] = rightward_trans[i]
            T_rightward[:3, :3] = T.get_rotation_matrix_from_xyz((0, 0, rads[i]))
            cam_pose = T_base @ T_rightward @ lidar2cam
            poses.append(cam_pose)
        return poses

    forward_front_cams = _generate_cam_poses(forward_trans, lidar2front_cam, front_rads)
    center_front_cams = _generate_cam_poses(0.5, lidar2front_cam, front_rads)
    center_back_cams = _generate_cam_poses(-0.5, lidar2back_cam, back_rads)
    backward_back_cams = _generate_cam_poses(-forward_trans, lidar2back_cam, back_rads)
    # cam_poses = forward_front_cams + center_front_cams + center_back_cams + backward_back_cams
    # cam_poses = forward_front_cams + backward_back_cams
    cam_poses = center_front_cams + center_back_cams
    cam_poses = np.stack(cam_poses)
    return cam_poses


def visualize_cam_poses(cam_poses, cam_K=None, img_size=None):
    lidar_frame = T.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    frames = [lidar_frame]
    for cam_pose in cam_poses:
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        cam_frame.transform(cam_pose)
        frames.append(cam_frame)
    # o3d.visualization.draw_geometries(frames)

    if cam_K is None:
        cam_K = NOVEL_CAM_K
            
    if img_size is None:
        img_size = NOVEL_IMG_SIZE

    fig = pv.figure()
    # ax = None
    for pose in cam_poses:
        fig.plot_transform(A2B=pose, s=0.5)
        fig.plot_camera(M=cam_K, cam2world=pose, virtual_image_distance=1, sensor_size=img_size)
        # ax = pc.plot_camera(ax=ax, M=cam_K, cam2world=pose, virtual_image_distance=1, sensor_size=img_size, ax_s=0.5, color='red')
    # plt.show()
    fig.show()


def generate_novel_depth(model, data_loader, save_root):
    model.eval()
    model.module.set_inference_mode(True)
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    save_dir = osp.join(save_root, 'novel_views')
    create_novel_views_dir(save_dir)

    novel_cam_poses = torch.tensor(sample_novel_cam_poses()).float().cuda()
    novel_cam_K = torch.tensor(NOVEL_CAM_K).float().cuda()
    for i, data in enumerate(data_loader):
        data['novel_cam_intrinsics'] = novel_cam_K
        data['novel_img_size'] = NOVEL_IMG_SIZE
        data['novel_cam_poses'] = novel_cam_poses
        with torch.no_grad():
            result = model(**data)

        visualize_novel_views(data, result, save_dir, i * world_size + rank)

        results.extend(result)

        if rank == 0:
            batch_size = data['img'].data[0].shape[0]
            for _ in range(batch_size * world_size):
                prog_bar.update()
    return results


if __name__ == '__main__':
    cam_poses = sample_novel_cam_poses()
    visualize_cam_poses(cam_poses)
