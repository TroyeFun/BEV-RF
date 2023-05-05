RUN=$1
torchrun --nproc_per_node 1 tools/test_nerf.py $RUN/configs.yaml $RUN/latest.pth --show --novel_view_inference
