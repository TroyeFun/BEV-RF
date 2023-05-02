import os
import os.path as osp

import mmcv
import torch

from test_utils import visualize_results


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def nerf_single_gpu_test(model, data_loader, save_dir):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    os.makedirs(osp.join(save_dir, 'vis'), exist_ok=True)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(**data)

        if i < 1000:
            vis_path = osp.join(save_dir, 'vis', f'{i}.png')
            visualize_results(data, result, vis_path)
        results.extend(result)

        batch_size = data['img'].data[0].shape[0]
        for _ in range(batch_size):
            prog_bar.update()
    return results
