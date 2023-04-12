import pickle

import ipdb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
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


def save_imgs(name, img):
    plt.imsave('tmp_data/batch0/images/' + name + '.jpg', img)


def visualize_data_batch0():
    imgs = np.load('tmp_data/batch0/img.npy')
    points = [
        np.load('tmp_data/batch0/points0.npy').astype(np.float32),
        np.load('tmp_data/batch0/points1.npy').astype(np.float32)]
    camera_intrinsics = np.load('tmp_data/batch0/camera_intrinsics.npy')
    img_aug_matrix = np.load('tmp_data/batch0/img_aug_matrix.npy')
    lidar2camera = np.load('tmp_data/batch0/lidar2camera.npy')
    lidar_aug_matrix = np.load('tmp_data/batch0/lidar_aug_matrix.npy')
    # metas = pickle.load(open('tmp_data/batch0/metas.pkl', 'rb'))
    source_cam2input_lidars = np.load('tmp_data/batch0/source_cam2input_lidars.npy')
    source_cam2target_cams = np.load('tmp_data/batch0/source_cam2target_cams.npy')
    source_camera_intrinsics = np.load('tmp_data/batch0/source_camera_intrinsics.npy')
    source_imgs = np.load('tmp_data/batch0/source_imgs.npy')
    target_imgs = np.load('tmp_data/batch0/target_imgs.npy')
    bev_feature = np.load('tmp_data/batch0/bev_feature.npy')

    imgs = imgs.transpose(0, 1, 3, 4, 2).astype(np.float32)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    source_imgs = source_imgs.transpose(0, 1, 3, 4, 2)
    target_imgs = target_imgs.transpose(0, 1, 3, 4, 2)

    for index in range(2):
        # save images
        for img_index in range(source_imgs[index].shape[0]):
            save_imgs('source_img_{}_{}'.format(index, img_index), source_imgs[index][img_index])
            save_imgs('target_img_{}_{}'.format(index, img_index), target_imgs[index][img_index])

        for img_index in range(imgs[index].shape[0]):
            save_imgs('img_{}_{}'.format(index, img_index), imgs[index][img_index])

        # project lidar to image
        for img_index in range(source_imgs[index].shape[0]):
            test_utils.project_lidar_to_image(
                'batch0/lidar2camera/source_{}_{}'.format(index, img_index),
                points[index][:, :3], source_imgs[index][img_index],
                np.linalg.inv(source_cam2input_lidars[index][img_index]),
                source_camera_intrinsics[index][img_index][:3, :3])
            lidar2target_cam = (source_cam2target_cams[index][img_index] @
                                np.linalg.inv(source_cam2input_lidars[index][img_index]))
            test_utils.project_lidar_to_image(
                'batch0/lidar2camera/target_{}_{}'.format(index, img_index),
                points[index][:, :3], target_imgs[index][img_index],
                lidar2target_cam,
                source_camera_intrinsics[index][img_index][:3, :3])

        for img_index in range(imgs[index].shape[0]):
            test_utils.project_lidar_to_image(
                'batch0/lidar2camera/input_{}_{}'.format(index, img_index),
                points[index][:, :3], imgs[index][img_index],
                lidar2camera[index][img_index],
                camera_intrinsics[index][img_index][:3, :3],
                lidar_aug_matrix[index], img_aug_matrix[index][img_index])

        pcd = test_utils.get_pcd(points[index][:, :3])
        lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        geometries = [pcd, lidar_frame]
        source_cam_frames = []
        target_cam_frames = []
        for cam_index in range(2):
            source_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            source_cam_frame.transform(source_cam2input_lidars[index][cam_index])
            source_cam_frames.append(source_cam_frame)
            target_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            target_cam_frame.transform(source_cam2input_lidars[index][cam_index] @ 
                                       np.linalg.inv(source_cam2target_cams[index][cam_index]))
            target_cam_frames.append(target_cam_frame)

        bev = test_utils.pcd_to_bev(points[index])
        save_imgs('bev_{}'.format(index), bev)
        bev_feature = bev_feature[0].max(0)
        save_imgs('bev_feature_{}'.format(index), bev_feature)

        o3d.visualization.draw_geometries(geometries)
        ipdb.set_trace()


if __name__ == '__main__':
    # show_cam_pts_and_voxel_feature()
    # modify_ckpt_vtransform_nx()
    visualize_data_batch0()