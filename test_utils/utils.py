import os
import os.path as osp

import cv2
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


def dump_data_batch(img, points, lidar2camera, lidar2image, camera_intrinsics, img_aug_matrix,
                    lidar_aug_matrix, source_imgs, target_imgs, source_camera_intrinsics,
                    source_cam2input_lidars, source_cam2target_cams, metas, bev_feature,
                    batch_id='0'):
    save_dir = 'batch{}/'.format(batch_id)
    os.makedirs('tmp_data/' + save_dir, exist_ok=True)
    dump_tensor(save_dir + 'img', img)
    for i, pcd in enumerate(points):
        dump_tensor(save_dir + 'points{}'.format(i), pcd)
    dump_tensor(save_dir + 'lidar2camera', lidar2camera)
    dump_tensor(save_dir + 'lidar2image', lidar2image)
    dump_tensor(save_dir + 'camera_intrinsics', camera_intrinsics)
    dump_tensor(save_dir + 'img_aug_matrix', img_aug_matrix)
    dump_tensor(save_dir + 'lidar_aug_matrix', lidar_aug_matrix)
    dump_tensor(save_dir + 'source_imgs', torch.cat(source_imgs))
    dump_tensor(save_dir + 'target_imgs', torch.cat(target_imgs))
    dump_tensor(save_dir + 'source_camera_intrinsics', torch.cat(source_camera_intrinsics))
    dump_tensor(save_dir + 'source_cam2input_lidars', torch.cat(source_cam2input_lidars))
    dump_tensor(save_dir + 'source_cam2target_cams', torch.cat(source_cam2target_cams))
    np.save('tmp_data/' + save_dir + 'metas.npy', np.asarray(str(metas)))
    dump_tensor(save_dir + 'bev_feature', bev_feature)


def project_lidar_to_image(save_name, pcd, image, lidar2camera, camera_intrinsics,
                           lidar_aug_matrix=None, img_aug_matrix=None):
    """
    Args:
        pcd (N, 3)
        K (3, 3)
        image (H, W, 3)
    Returns:
        pix (N, 2)
    """
    if lidar_aug_matrix is not None:
        lidar_aug_matrix_inv = np.linalg.inv(lidar_aug_matrix)
        pcd = pcd @ lidar_aug_matrix_inv[:3, :3] + lidar_aug_matrix_inv[:3, 3:4].T
    pcd = pcd @ lidar2camera[:3, :3].T + lidar2camera[:3, 3:4].T
    pcd = pcd[pcd[:, 2] > 0]
    depth = pcd[:, 2].copy()
    pcd /= pcd[:, 2:3]
    pix = pcd @ camera_intrinsics.T
    if img_aug_matrix is not None:
        pix[:, 2] = depth
        pix = pix @ img_aug_matrix[:3, :3].T + img_aug_matrix[:3, 3:4].T
    mask = (pix[:, 0] >= 0) & (pix[:, 0] < image.shape[1]) & (pix[:, 1] >= 0) & (pix[:, 1] < image.shape[0])
    pix = pix[mask]
    depth = depth[mask]

    plt.clf()
    plt.imshow(image)
    plt.scatter(pix[:, 0], pix[:, 1], c=depth, cmap='rainbow_r', alpha=0.2, s=0.5)
    plt.savefig('tmp_data/' + save_name + '.png')


def pcd_to_bev(pcd, voxel_size=(0.2, 0.2), scene_range=(-25.6, -51.2, 25.6, 51.2)):
    """Convert point cloud to bev.
    Returns:
        np.ndarray: bev.
    """
    mask = ((pcd[:, 0] >= scene_range[0]) * (pcd[:, 0] < scene_range[2]) *
            (pcd[:, 1] >= scene_range[1]) * (pcd[:, 1] < scene_range[3]))
    pcd = pcd[mask]
    coords = np.zeros_like(pcd[:, :2])
    coords[:, 0] = (pcd[:, 0] - scene_range[0]) // voxel_size[0]
    coords[:, 1] = (pcd[:, 1] - scene_range[1]) // voxel_size[1]
    bev = np.zeros((int((scene_range[2] - scene_range[0]) / voxel_size[0]),
                    int((scene_range[3] - scene_range[1]) / voxel_size[1])))
    bev[coords[:, 0].astype(np.int64), coords[:, 1].astype(np.int64)] = 1
    return bev


def visualize_results(input_batch, result, save_path):
    input_imgs = input_batch['img'].data[0]
    input_imgs = (input_imgs - input_imgs.min()) / (input_imgs.max() - input_imgs.min())
    source_imgs = input_batch['source_imgs'].data[0]
    color_rendered = result['color_rendered']
    depth_rendered = result['depth_rendered']
    color_sampled = result['color_sampled']
    bs = len(source_imgs)
    n_sources, n_cams = source_imgs[0].shape[:2]
    n_imgs = bs * n_sources * n_cams
    n_cols = 4

    plt.clf()
    fig = plt.figure(figsize=(10 * n_cols, 4 * n_imgs), dpi=72)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1)
    for bid in range(bs):
        for sid in range(n_sources):
            for cid in range(n_cams):
                img_id = bid * n_sources * n_cams + sid * n_cams + cid
                ax0 = fig.add_subplot(n_imgs, n_cols, img_id * n_cols + 1)
                ax0.imshow(input_imgs[bid][cid].permute(1, 2, 0).cpu().numpy())
                ax1 = fig.add_subplot(n_imgs, n_cols, img_id * n_cols + 2)
                ax1.imshow(source_imgs[bid][sid][cid].permute(1, 2, 0).cpu().numpy())
                ax2 = fig.add_subplot(n_imgs, n_cols, img_id * n_cols + 3)
                ax2.imshow(color_rendered[bid][sid][cid].cpu().numpy())
                ax3 = fig.add_subplot(n_imgs, n_cols, img_id * n_cols + 4)
                ax3.imshow(depth_rendered[bid][sid][cid].cpu().numpy(), cmap='rainbow_r')
    plt.savefig(save_path)
    # plt.show()


def create_novel_views_dir(save_dir):
    vis_dir = osp.join(save_dir, 'vis')
    color_dir = osp.join(save_dir, 'color')
    depth_dir = osp.join(save_dir, 'depth')
    valid_ray_dir = osp.join(save_dir, 'valid_ray')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(valid_ray_dir, exist_ok=True)


def visualize_novel_views(input_batch, result, save_dir, index):
    input_imgs = input_batch['img'].data[0][0]
    input_imgs = (input_imgs - input_imgs.min()) / (input_imgs.max() - input_imgs.min())
    input_imgs = input_imgs.permute(0, 2, 3, 1).cpu().numpy()
    colors = result['colors'].cpu().numpy().clip(0.0, 1.0)
    depths = result['depths'].cpu().numpy()
    ray_valid_ratios = result['ray_valid_ratios'].cpu().numpy()

    n_input = input_imgs.shape[0]
    n_novel = colors.shape[0]
    n_cols = 3
    n_rows = 2 + n_novel

    plt.clf()
    fig = plt.figure(figsize=(10 * n_cols, 4 * n_rows), dpi=72)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1)
    for i in range(n_input):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(input_imgs[i])

    for i in range(n_novel):
        ax = fig.add_subplot(n_rows, n_cols, 6 + i * 3 + 1)
        ax.imshow(colors[i])
        ax = fig.add_subplot(n_rows, n_cols, 6 + i * 3 + 2)
        ax.imshow(depths[i], cmap='rainbow_r')
        ax = fig.add_subplot(n_rows, n_cols, 6 + i * 3 + 3)
        ax.imshow(ray_valid_ratios[i], cmap='rainbow_r')
        plt.imsave(osp.join(save_dir, 'color', f'{index:05d}_{i:02d}.png'), colors[i])
        np.save(osp.join(save_dir, 'depth', f'{index:05d}_{i:02d}.npy'), depths[i])
        np.save(osp.join(save_dir, 'valid_ray', f'{index:05d}_{i:02d}.npy'), ray_valid_ratios[i])

    plt.savefig(osp.join(save_dir, 'vis', f'{index:05d}.png'))
    # plt.show()