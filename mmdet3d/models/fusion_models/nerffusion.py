from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["NerfFusion"]


@FUSIONMODELS.register_module()
class NerfFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()
        self._inference_mode = kwargs.get('novel_view_inference', False)

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        imgs = x.view(B * N, C, H, W)  # (6, 3, 256, 704), bs=1

        x = self.encoders["camera"]["backbone"](imgs)  # [(6, 192, 32, 88), (6, 384, 16, 44), (6, 768, 8, 22)]
        x = self.encoders["camera"]["neck"](x) # [(6, 256, 32, 88), (6, 256, 16, 44), (6, 256, 8, 22)]

        # upsample features to original image size and concat 
        cam_feat_h, cam_feat_w = x[0].size(2), x[0].size(3)
        x = [imgs] + x
        for i in range(len(x)):
            x[i] = F.interpolate(x[i], size=(cam_feat_h, cam_feat_w),
                                 **self.encoders["camera"]["neck"].upsample_cfg)
        x = torch.cat(x, dim=1)  # (6, 3 + 256 * 3, 32, 88)

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def set_inference_mode(self, mode=True):
        self._inference_mode = mode
        self.heads['nerf'].set_inference_mode(mode)

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        source_imgs,
        target_imgs,
        source_camera_intrinsics,
        source_cam2input_lidars,
        source_cam2target_cams,
        metas,
        **kwargs,
    ):
        if self._inference_mode:
            assert not self.training
            outputs = self.inference_novel_views(
                img,
                points,
                lidar2camera,
                camera_intrinsics,
                kwargs["novel_cam_intrinsics"],
                kwargs["novel_img_size"],
                kwargs["novel_cam_poses"],
            )
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                source_imgs,
                target_imgs,
                source_camera_intrinsics,
                source_cam2input_lidars,
                source_cam2target_cams,
                metas,
                **kwargs,
            )
        return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        source_imgs,
        target_imgs,
        source_camera_intrinsics,
        source_cam2input_lidars,
        source_cam2target_cams,
        metas,
    ):
        cam_feat = None
        lidar_feat = None
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]  # avoid OOM
        ):  # camera, lidar
            if sensor == "camera":
                # (bs, 3 + 256 * 3, 256, 704)
                cam_feat = self.extract_camera_features(img)
            elif sensor == "lidar":
                # (bs, 256, 128, 128)
                lidar_feat = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

        bev_feat = self.decoder["backbone"](lidar_feat)  # [(bs, 128, 128, 128), (bs, 256, 64, 64))]
        bev_feat = self.decoder["neck"](bev_feat)  # [(bs, 512, 64, 128)]

        if self.training:
            outputs = {}
            losses = self.heads["nerf"](
                cam_feat=cam_feat, bev_feat=bev_feat,
                source_imgs=source_imgs, target_imgs=target_imgs,
                raw_cam_Ks=camera_intrinsics, source_cam_Ks=source_camera_intrinsics,
                lidar2cams=lidar2camera, source_cam2input_lidars=source_cam2input_lidars,
                source_cam2target_cams=source_cam2target_cams, points=points)
            for name, val in losses.items():
                outputs[f"stats/nerf/{name}"] = val
            return outputs
        else:
            outputs = self.heads['nerf'](
                cam_feat=cam_feat, bev_feat=bev_feat,
                source_imgs=source_imgs, target_imgs=target_imgs,
                raw_cam_Ks=camera_intrinsics, source_cam_Ks=source_camera_intrinsics,
                lidar2cams=lidar2camera, source_cam2input_lidars=source_cam2input_lidars,
                source_cam2target_cams=source_cam2target_cams, points=points)
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def inference_novel_views(
        self,
        img,
        points,
        lidar2camera,
        camera_intrinsics,
        novel_cam_K,  # [3, 3]
        novel_img_size,  # (x, y)
        novel_cam_poses,  # (n_novel_cams, 4, 4)
    ):
        cam_feat = None
        lidar_feat = None
        for sensor in list(self.encoders.keys())[::-1]:  # lidar, camera, avoid OOM
            if sensor == "camera":
                # (bs, 3 + 256 * 3, 256, 704)
                cam_feat = self.extract_camera_features(img)
            elif sensor == "lidar":
                # (bs, 256, 128, 128)
                lidar_feat = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

        bev_feat = self.decoder["backbone"](lidar_feat)  # [(bs, 128, 128, 128), (bs, 256, 64, 64))]
        bev_feat = self.decoder["neck"](bev_feat)  # [(bs, 512, 64, 128)]
        outputs = self.heads["nerf"].inference_novel_views(
            cam_feat=cam_feat, bev_feat=bev_feat,
            raw_cam_Ks=camera_intrinsics, lidar2cams=lidar2camera,
            novel_cam_K=novel_cam_K, novel_img_size=novel_img_size, novel_cam_poses=novel_cam_poses)
        return outputs