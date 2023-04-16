RUN=$1
torchpack dist-run -np 1 python tools/test_nerf.py $RUN/configs.yaml $RUN/latest.pth --show
