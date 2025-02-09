DATE=`date +%y%m%d`
# torchpack dist-run -np 1 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
# torchpack dist-run -np 1 python tools/train.py configs/bevnerf_nuscenes/bevnerf/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
# torchpack dist-run -np 2 python tools/train.py configs/bevnerf_nuscenes/bevnerf/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate
# torchpack dist-run -np 2 python tools/train.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate
# torchrun --nproc_per_node=2 tools/train.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate  --run-dir runs/test
# torchrun --nproc_per_node=4 tools/train_torch_dist.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate
# torchrun --nproc_per_node 1 tools/train.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate --run-dir runs/run-nerffusion-source_diff_to_input-max_seq_dist_6
# torchpack dist-run -np 1 python tools/train.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate --run-dir runs/run-nerffusion-source_diff_to_input-max_seq_dist_6
torchrun --nproc_per_node 1 tools/train.py configs/nerffusion_nuscenes/nerffusion/fusion-bev256d2-lss.yaml --load_from pretrained/bevfusion-seg-half_x_range.pth --no-validate --run-dir runs/$DATE-nerffusion-max_seq_dist_6-kl1-dist2closest0.01-unet
