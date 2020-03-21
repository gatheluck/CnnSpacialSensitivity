import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import torch
import torchvision

from spatial_sensitivity.shuffle import shuffle

class PatchShuffle(object):
    def __init__(self, num_h_partition:int, num_w_partition:int):
        """
        apply Patch Shuffle to input tensor along height and width dimension.   

        Args
        - num_h_partition: number of partition about hight dimension
        - num_w_partition: number of partition about width dimension 
        """
        assert num_h_partition >= 0
        assert num_w_partition >= 0
        self.num_h_partition = num_h_partition
        self.num_w_partition = num_w_partition

    def __call__(self, x:torch.tensor):
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= self.num_h_partition and h>0
        assert w >= self.num_w_partition and w>0

        x = shuffle(x, -2, self.num_h_partition)
        x = shuffle(x, -1, self.num_w_partition)

        return x

if __name__ == '__main__':
    import tqdm

    x_list=[]
    os.makedirs('../logs', exist_ok=True)
    for h_idx in tqdm.tqdm(range(5)):
        for w_idx in range(5):
            transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            PatchShuffle(h_idx,w_idx),
                        ])
            dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, transform=transform, download=True)
            loader  = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)


            for i, (x,t) in enumerate(loader):
                x_list.append(x[1])
                break

    torchvision.utils.save_image(torch.stack(x_list, dim=0), "../logs/patch_shuffle.png", nrow=5, padding=1)