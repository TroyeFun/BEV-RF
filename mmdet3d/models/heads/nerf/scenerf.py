# -*- coding:utf-8 -*-
# Copyright @ 2023 Peking University. All rights reserved.
# Authors: fanghongyu (fanghongyu@pku.edu.cn)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import HEADS
from mmdet3d.models.heads.nerf.utils import (PositionalEncoding, ResnetFC, RaySOM,
    sample_pix_from_img, compute_direction_from_pixels, sample_rays_viewdir, sample_rays_gaussian,
    cam_pts_2_cam_pts, cam_pts_2_pix, sample_feats_3d, pix_2_cam_pts)


__all__ = ["SceneRFHead"]


def compute_l1_loss(pred, target, predicted_depth=None):
    """
    pred: (3, B)
    target: (3, B)
    ---
    return
    l1_loss: (B,)

    """
    abs_diff = torch.abs(target - pred)
    if predicted_depth is not None:
        abs_diff = abs_diff[:, predicted_depth < 30]
    l1_loss = abs_diff.mean(0)

    return l1_loss


@HEADS.register_module()
class SceneRFHead(nn.Module):
    """_summary_

                   W
        -----------------------
        |                     |
        |                     |
        |           ---->y    | H
        |          |          |
        |          v x        |
        -----------------------
    """

    def __init__(
            self,
            dim_voxel_feature=128,
            n_rays=1200,
            n_pts_uni=32,
            n_gaussians=4,
            n_pts_per_gaussian=8,
            gaussian_std=2.5,
            max_sample_depth=100,
            som_sigma=2.0,
            scene_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self._dim_voxel_feature = dim_voxel_feature

        self._n_rays = n_rays
        self._ray_batch_size = self._n_rays
        self._n_pts_uni = n_pts_uni
        self._n_gaussians = n_gaussians
        self._n_pts_per_gaussian = n_pts_per_gaussian
        self._gaussian_std = gaussian_std
        self._max_sample_depth = max_sample_depth
        self._scene_range = np.array(scene_range)

        self._positional_encoding = PositionalEncoding(
            num_freqs=6, include_input=True)
        self._mlp = ResnetFC(
            d_in=39 + 3,  # positional_encoding + ray_direction
            d_out=4,
            n_blocks=3,
            d_hidden=128,
            d_latent=dim_voxel_feature,
        )
        self._mlp_gaussian = ResnetFC(
            d_in=39 + 3,
            d_out=2,
            n_blocks=3,
            d_hidden=128,
            d_latent=dim_voxel_feature,
        )
        self._ray_som = RaySOM(som_sigma=som_sigma)

    def forward(self, batch, step_type):
        """
        Args:
            batch (dict): {
                'bev_feature': (B, C, H, W)
                'source_camera_intrinsics': camera_intrinsics, List[B * (n_sources, n_cam, 4, 4)]
                'source_imgs': List[B * (n_sources, n_cam, 3, 900, 1600)]
                'target_imgs': List[B * (n_sources, n_cam, 3, 900, 1600)]
                'source_cam2input_lidars': List[B * (n_sources, n_cam, 4, 4)]
                'source_cam2target_cams': List[B * (n_sources, n_cam, 4, 4)]
            }
        """
        bev_feature = batch['bev_feature']
        B, C, H, W = bev_feature.shape
        voxel_feature = bev_feature.reshape(
            B, self._dim_voxel_feature, C // self._dim_voxel_feature, H, W) # (B, C', D, H, W)

        total_loss_kl = 0
        total_loss_dist2closest = 0
        total_loss_reprojection = 0
        total_loss_color = 0
        total_min_stds = 0
        total_min_som_vars = 0

        for i in range(B):
            cam_K = batch['source_camera_intrinsics'][i]
            inv_K = torch.inverse(cam_K)
            source2targets = batch['source_cam2target_cams'][i]
            source2inputs = batch['source_cam2input_lidars'][i]
            source_imgs = batch['source_imgs'][i]
            target_imgs = batch['target_imgs'][i]
            n_sources = len(source_imgs)
            n_cams = len(source_imgs[0])

            for sid in range(n_sources):
                for cam_id in range(n_cams):
                    source_img = source_imgs[sid][cam_id]
                    target_img = target_imgs[sid][cam_id]
                    source2input = source2inputs[sid][cam_id]
                    source2target = source2targets[sid][cam_id]
                    ret = self._process_single_source(
                        voxel_feature[i], cam_K=cam_K[sid][cam_id], inv_K=inv_K[sid][cam_id],
                        source_img=source_img, target_img=target_img,
                        source2target=source2target, source2input=source2input)
                    total_loss_kl += ret['loss_kl'].mean()
                    total_loss_dist2closest += ret['loss_dist2closest'].mean()
                    total_loss_reprojection += ret['loss_reprojection'].mean()
                    total_loss_color += ret['loss_color'].mean()
                    total_min_stds += ret['min_stds'].mean()
                    total_min_som_vars += ret['min_som_vars'].mean()
                    # TODO: evaluate depth

            total_loss = 0
            if self.use_reprojection:
                total_loss += total_loss_reprojection

            if self.use_color:
                total_loss += total_loss_color

            total_loss += total_loss_kl
            total_loss += total_loss_dist2closest * 0.01
            total_loss /= B
            return {
                'total_loss': total_loss
            }


    def _process_single_source(self, voxel_feature, cam_K, inv_K, source_img, target_img,
                               source2target, source2input):
        xs = torch.arange(start=0, end=source_img.shape[1], step=2).type_as(cam_K)
        ys = torch.arange(start=0, end=source_img.shape[2], step=2).type_as(cam_K)
        grid_x, grid_y = torch.meshgrid(xs, ys)
        sampled_pixels = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1)

        perm = torch.randperm(sampled_pixels.shape[0])
        sampled_pixels = sampled_pixels[perm[:self._n_rays]]
        render_out_dict = self._render_rays_batch(
            inv_K, source_img.shape[1:3], source2input, voxel_feature, sampled_pixels=sampled_pixels)

        depth_source_rendered = render_out_dict['depth']
        color_rendered = render_out_dict['color']
        gaussian_means = render_out_dict['gaussian_means']
        gaussian_stds = render_out_dict['gaussian_stds']
        som_vars = render_out_dict['som_vars']
        loss_kl = render_out_dict['loss_kl']

        weights_at_depth = render_out_dict['weights_at_depth']
        # closest_pts_to_depths = render_out_dict['closest_pts_to_depths']

        # loss
        diff = torch.abs(gaussian_means - depth_source_rendered.unsqueeze(-1).detach())
        min_diff, gaussian_idx = torch.min(diff, dim=1)
        loss_dist2closest = min_diff

        min_stds = torch.gather(gaussian_stds, 1, gaussian_idx.unsqueeze(-1))
        min_som_vars = torch.gather(som_vars, 1, gaussian_idx.unsqueeze(-1))

        sampled_color_source = sample_pix_from_img(source_img, sampled_pixels)
        loss_color = torch.abs(sampled_color_source.T - color_rendered)

        loss_reprojection = self._compute_reprojection_loss(
            sampled_pixels, sampled_color_source, depth_source_rendered,
            target_img, inv_K, cam_K, source2target)

        ret = {
            'loss_kl': loss_kl,
            'loss_dist2closest': loss_dist2closest,
            'loss_reprojection': loss_reprojection,
            'loss_color': loss_color,
            'weights_at_depth': weights_at_depth,
            'min_som_vars': min_som_vars,
            'min_stds': min_stds,
        }
        return ret

    def _render_rays_batch(self, inv_K, img_size, source2input, voxel_feature, sampled_pixels):
        depth_rendereds = []
        gaussian_means = []
        gaussian_stds = []
        weights_at_depth = []
        closest_pts_to_depths = []
        som_vars = []
        densities = []
        weights = []
        alphas = []
        depth_volumes = []
        color_rendereds = []

        cnt = 0
        loss_kl = []

        for start_i in range(0, sampled_pixels.shape[0], self._ray_batch_size):
            end_i = min(start_i + self._ray_batch_size, sampled_pixels.shape[0])
            sampled_pixels_batch = sampled_pixels[start_i:end_i]
            ret = self._batchify_depth_and_color(
                source2input, voxel_feature, sampled_pixels_batch, inv_K, img_size)
            color_rendereds.append(ret['color'])
            depth_rendereds.append(ret['depth'])
            gaussian_means.append(ret['gaussian_means'])
            gaussian_stds.append(ret['gaussian_stds'])
            weights_at_depth.append(ret['weights_at_depth'].reshape(-1))
            closest_pts_to_depths.append(ret['closest_pts_to_depth'])
            loss_kl.append(ret['loss_kl'])
            densities.append(ret['density'])
            weights.append(ret['weights'])
            alphas.append(ret['alphas'])
            depth_volumes.append(ret['depth_volume'])
            som_vars.append(ret['som_vars'])
            cnt += 1

        depth_rendereds = torch.cat(depth_rendereds, dim=0)
        gaussian_means = torch.cat(gaussian_means, dim=0)
        gaussian_stds = torch.cat(gaussian_stds, dim=0)
        weights_at_depth = torch.cat(weights_at_depth, dim=0)
        closest_pts_to_depths = torch.cat(closest_pts_to_depths, dim=0)
        loss_kl = torch.cat(loss_kl, dim=0)
        densities = torch.cat(densities, dim=0)
        weights = torch.cat(weights, dim=0)
        alphas = torch.cat(alphas, dim=0)
        depth_volumes = torch.cat(depth_volumes, dim=0)
        som_vars = torch.cat(som_vars, dim=0)
        color_rendereds = torch.cat(color_rendereds, dim=0)
        ret = {
            'depth': depth_rendereds,
            'color': color_rendereds,
            'gaussian_means': gaussian_means,
            'gaussian_stds': gaussian_stds,
            'weights_at_depth': weights_at_depth,
            'closest_pts_to_depths': closest_pts_to_depths,
            'loss_kl': loss_kl,
            'densities': densities,
            'weights': weights,
            'alphas': alphas,
            'depth_volumes': depth_volumes,
            'som_vars': som_vars,
        }
        return ret

    def _compute_reprojection_loss(self, pix_source, sampled_color_source, depth_rendered,
                                  img_target, inv_K, cam_K, source2target):
        cam_source_pts = pix_2_cam_pts(pix_source, inv_K, depth_rendered)
        cam_pts_target = cam_pts_2_cam_pts(cam_source_pts, source2target)
        pix_target = cam_pts_2_pix(cam_pts_target, cam_K)
        mask = cam_pts_target[:, 2] > 0
        pix_source = pix_source[mask, :]
        pix_target = pix_target[mask, :]
        sampled_color_source = sampled_color_source[:, mask]

        sampled_color_target = sample_pix_from_img(pix_target, img_target)
        sampled_color_target_identity_reprojection = sample_pix_from_img(pix_source, img_target)
        loss_reprojection = compute_l1_loss(sampled_color_source, sampled_color_target)
        loss_indentity_reprojection = compute_l1_loss(
            sampled_color_source, sampled_color_target_identity_reprojection)
        loss_indentity_reprojection += torch.randn(loss_indentity_reprojection.shape).cuda() * 0.00001
        loss_reprojections = torch.stack([loss_reprojection, loss_indentity_reprojection]).min(
            dim=0)[0]
        return loss_reprojections

    def _batchify_depth_and_color(self, source2input, voxel_feature, sampled_pixels_batch,
                                  inv_K, img_size):
        ret = {}
        n_rays = sampled_pixels_batch.shape[0]
        unit_direction = compute_direction_from_pixels(sampled_pixels_batch, inv_K)
        cam_pts_uni, depth_volume_uni, sensor_distance_uni, viewdir = sample_rays_viewdir(
            inv_K, source2input, img_size,
            sampled_methods='uniform',
            sampled_pixels=sampled_pixels_batch,
            n_pts_per_ray=self._n_pts_uni,
            max_sample_depth=self._max_sample_depth)
        (gaussian_means_sensor_distance, gaussian_stds_sensor_distance
        ) = self._predict_gaussian_means_and_stds(
            source2input, unit_direction, voxel_feature, viewdir)
        cam_pts_gaussian, depth_volume_gaussian, sensor_distance_gaussian = sample_rays_gaussian(
            source2input, n_rays, unit_direction, gaussian_means_sensor_distance,
            gaussian_stds_sensor_distance, self._max_sample_depth, self._n_gaussians,
            self._n_pts_per_gaussian)

        cam_pts = torch.cat([cam_pts_uni, cam_pts_gaussian], dim=1)
        depth_volume = torch.cat([depth_volume_uni, depth_volume_gaussian], dim=1)
        sensor_distance = torch.cat([sensor_distance_uni, sensor_distance_gaussian], dim=1)

        sorted_indices = torch.argsort(sensor_distance, dim=1)
        sensor_distance = torch.gather(sensor_distance, 1, sorted_indices)
        cam_pts = torch.gather(cam_pts, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3))
        depth_volume = torch.gather(depth_volume, 1, sorted_indices)

        densities, colors = self._predict(self._mlp, cam_pts.detach(), viewdir, voxel_feature)
        rendered_out = self._render_depth_and_color(densities, colors, sensor_distance, depth_volume)
        loss_kl, som_means, som_vars = self._ray_som(
            gaussian_means_sensor_distance, gaussian_stds_sensor_distance, sensor_distance,
            rendered_out['alphas'])

        ret['color'] = rendered_out['color']
        ret['depth'] = rendered_out['depth_rendered']
        ret['gaussian_means'] = gaussian_means_sensor_distance
        ret['gaussian_stds'] = gaussian_stds_sensor_distance
        ret['weights_at_depth'] = rendered_out['weights_at_depth']
        ret['closest_pts_to_depth'] = rendered_out['closest_pts_to_depth']
        ret['loss_kl'] = loss_kl
        ret['som_vars'] = som_vars
        ret['density'] = densities
        ret['weights'] = rendered_out['weights']
        ret['alphas'] = rendered_out['alphas']
        ret['depth_volume'] = depth_volume
        return ret

    def _predict_gaussian_means_and_stds(self, source2input, unit_direction, voxel_feature, viewdir):
        n_rays = unit_direction.shape[0]
        step = self._max_sample_depth * 1.0 / self._n_gaussians
        gaussian_means_sensor_distance = torch.linspace(step / 2, self._max_sample_depth - step / 2,
                                                        self._n_gaussians).type_as(unit_direction)
        gaussian_means_sensor_distance = gaussian_means_sensor_distance.view(
            1, self._n_gaussians, 1).expand(n_rays, -1, 1)
        direction = unit_direction.view(n_rays, 1, 3).expand(-1, self._n_gaussians, -1)
        gaussian_means_pts = gaussian_means_sensor_distance * direction
        gaussian_means_pts_infer = cam_pts_2_cam_pts(gaussian_means_pts.reshape(-1, 3),
                                                     source2input)
        gaussian_means_pts_infer = gaussian_means_pts_infer.reshape(n_rays, self._n_gaussians, 3)
        output = self._predict(self._mlp_gaussian, gaussian_means_pts_infer, viewdir, voxel_feature,
                               output_type='offset')
        gaussian_means_offset = output[:, :, 0]
        gaussian_stds_offset = output[:, :, 1]
        gaussian_means_sensor_distance = gaussian_means_sensor_distance.squeeze(
            -1) + gaussian_means_offset
        gaussian_means_sensor_distance = torch.relu(gaussian_means_sensor_distance) + 1.5
        gaussian_stds_sensor_distance = torch.relu(gaussian_stds_offset + self._gaussian_std) + 1.5
        return gaussian_means_sensor_distance, gaussian_stds_sensor_distance

    def _predict(self, mlp, cam_pts, viewdir, voxel_feature, output_type='density'):
        saved_shape = cam_pts.shape
        cam_pts = cam_pts.reshape(-1, 3)
        pe = self._positional_encoding(cam_pts)
        pts_feature = sample_feats_3d(voxel_feature, cam_pts, self._scene_range)
        viewdir = viewdir.unsqueeze(1).expand(-1, saved_shape[1], -1).reshape(-1, 3)
        x_in = torch.cat([pts_feature, pe, viewdir], dim=-1)

        if output_type == 'density':
            mlp_output = mlp(x_in)
            color = torch.sigmoid(mlp_output[:, :3]).view(saved_shape[0], saved_shape[1], 3)
            density = self._density_activation(mlp_output[:, 3:4]).view(
                saved_shape[0], saved_shape[1], 1)
            return density, color
        elif output_type == 'offset':
            mlp_output = mlp(x_in)
            residual = mlp_output.view(saved_shape[0], saved_shape[1], 2)
            return residual

    def _density_activation(self, density_logit):
        return F.softplus(density_logit - 1, beta=1)

    def _render_depth_and_color(self, densities, colors, sensor_distance, depth_volume):
        sensor_distance = sensor_distance.clone()
        sensor_distance[sensor_distance < 0] = 0
        deltas = torch.zeros_like(sensor_distance)
        deltas[:, 0] = sensor_distance[:, 0]
        deltas[:, 1:] = sensor_distance[:, 1:] - sensor_distance[:, :-1]
        alphas = 1 - torch.exp(-deltas * densities)

        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)
        T_alphas = torch.cumprod(alphas_shifted, dim=-1)
        weights = alphas * T_alphas[:, :-1]

        color_rendered = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        depth_rendered = torch.sum(weights * depth_volume, -1)
        abs_diff = torch.abs(depth_rendered.unsqueeze(-1) - depth_volume)
        closest_pts_to_depth, weights_at_depth_idx = torch.min(abs_diff, dim=1)
        weights_at_depth = torch.gather(weights, 1, weights_at_depth_idx.unsqueeze(-1)).squeeze()

        ret = {
            'alphas': alphas,
            'color': color_rendered,
            'depth_rendered': depth_rendered,
            'weights_at_depth': weights_at_depth,
            'closest_pts_to_depth': closest_pts_to_depth,
            'weights': weights,
        }
        return ret
