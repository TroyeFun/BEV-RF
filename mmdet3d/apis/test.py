import os
import os.path as osp
import time

import mmcv
from mmcv.runner import get_dist_info
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


def nerf_multi_gpu_test(model, data_loader, save_dir):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    os.makedirs(osp.join(save_dir, 'vis'), exist_ok=True)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(**data)
        
        vis_path = osp.join(save_dir, 'vis', f'{i * world_size + rank}.png')
        visualize_results(data, result, vis_path)

        results.extend(result)

        if rank == 0:
            batch_size = data['img'].data[0].shape[0]
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    # if gpu_collect:
    #     results = collect_results_gpu(results, len(dataset))
    # else:
    #     results = collect_results_cpu(results, len(dataset), tmpdir)
    return results