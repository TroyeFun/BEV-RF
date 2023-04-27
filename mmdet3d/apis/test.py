import mmcv
import torch


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


def nerf_single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(**data)

        import ipdb; ipdb.set_trace()
        # import test_utils; test_utils.visualize_results(data, result)
        results.extend(result)

        batch_size = data['img'].data[0].shape[0]
        for _ in range(batch_size):
            prog_bar.update()
    return results
