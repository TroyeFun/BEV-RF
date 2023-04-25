# torchpack dist-run -np 1 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
# torchpack dist-run -np 1 python tools/train.py configs/bevnerf_nuscenes/bevnerf/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
# torchpack dist-run -np 2 python tools/train.py configs/bevnerf_nuscenes/bevnerf/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate
torchpack dist-run -np 2 python tools/train.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate
torchrun --nproc_per_node=4 tools/train_torch_dist.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --
no-validate
