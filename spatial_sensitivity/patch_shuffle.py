import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import tqdm
import collections

import numpy as np
import torch
import torchvision

from misc.metric import get_num_correct
from misc.logger import Logger
from misc.plot import plot

from spatial_sensitivity.shuffle import shuffle


class PatchShuffle():
    def __init__(self, num_h_devide: int, num_w_devide: int):
        """
        apply Patch Shuffle to input tensor along height and width dimension.

        Args
        - num_h_devide: number of devision about hight dimension
        - num_w_devide: number of devision about width dimension
        """
        assert num_h_devide >= 0
        assert num_w_devide >= 0
        self.num_h_devide = num_h_devide
        self.num_w_devide = num_w_devide

    def __call__(self, x: torch.tensor):
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= self.num_h_devide and h > 0
        assert w >= self.num_w_devide and w > 0

        x = shuffle(x, -2, self.num_h_devide)
        x = shuffle(x, -1, self.num_w_devide)

        return x


def eval_patch_shuffle(model, dataset_builder, max_num_devide: int, num_samples: int, batch_size: int, num_workers: int, top_k: int, log_dir: str, log_params: dict = {}, suffix: str = '', shuffle: bool = False, **kwargs):
    """
    Args
    - model: NN model
    - dataset_builder: DatasetBuilder class object
    - max_num_devide: max number of division
    - num_samples: number of sample to use. if -1, all samples are used
    - batch_size: size of batch
    - num_workers: number of workers
    - top_k: use top_k accuracy
    - log_dir: log directory
    - log_params: params which is logged in dataframe. these params are useful for legend.
    - suffix: suffix of log
    - shuffle: shuffle data
    """
    assert max_num_devide >= 1
    assert num_samples >= 1 or num_samples == -1
    assert batch_size >= 1
    assert num_workers >= 1
    assert top_k >= 1

    log_path = os.path.join(log_dir, os.path.join('pathch_shuffle_result' + suffix + '.csv'))
    logger = Logger(path=log_path, mode='test')

    # log params
    # logger.log(log_params)

    acc_dict = {}
    images_list = []

    for num_devide in tqdm.tqdm(range(1, max_num_devide + 1)):
        log_dict = collections.OrderedDict()

        # build Patch Shuffled dataset
        patch_shuffle_transform = PatchShuffle(num_devide, num_devide)
        dataset = dataset_builder(train=False, normalize=True, optional_transform=[patch_shuffle_transform])
        if num_samples != -1:
            num_samples = min(num_samples, len(dataset))
            indices = [i for i in range(num_samples)]
            dataset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        with torch.autograd.no_grad():
            num_correct = 0.0
            for i, (x, t) in enumerate(loader):
                model.eval()
                x = x.to('cuda', non_blocking=True)
                t = t.to('cuda', non_blocking=True)

                model.zero_grad()
                logit = model(x)
                num_correct += get_num_correct(logit, t, topk=top_k)

                if i == 0:
                    images_list.append(x[10])

        acc = num_correct / float(len(dataset))
        key = '{num_devide}'.format(num_devide=num_devide)
        acc_dict[key] = acc

        log_dict['num_devide'] = num_devide
        log_dict['accuracy'] = acc
        logger.log(log_dict)
        print(acc_dict)

    # save data
    torch.save(acc_dict, os.path.join(log_dir, 'patch_shuffle_acc_dict' + suffix + '.pth'))
    torchvision.utils.save_image(torch.stack(images_list, dim=0), os.path.join(log_dir, 'example_images' + suffix + '.png'), nrow=max_num_devide)
    plot(csv_path=log_path, x='num_devide', y='accuracy', hue=None, log_path=os.path.join(log_dir, 'plot.png'), save=True)


if __name__ == '__main__':
    import tqdm

    x_list = []
    os.makedirs('../logs', exist_ok=True)
    for h_idx in tqdm.tqdm(range(1, 6)):
        for w_idx in range(1, 6):
            transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            PatchShuffle(h_idx, w_idx),
                        ])
            dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, transform=transform, download=True)
            loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

            for i, (x, t) in enumerate(loader):
                x_list.append(x[1])
                break

    torchvision.utils.save_image(torch.stack(x_list, dim=0), "../logs/patch_shuffle.png", nrow=5, padding=1)