import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import torch
import torchvision

from spatial_sensitivity.shuffle import shuffle

class PatchShuffle(object):
    def __init__(self, num_h_devide:int, num_w_devide:int):
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

    def __call__(self, x:torch.tensor):
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= self.num_h_devide and h>0
        assert w >= self.num_w_devide and w>0

        x = shuffle(x, -2, self.num_h_devide)
        x = shuffle(x, -1, self.num_w_devide)

        return x

if __name__ == '__main__':
    import tqdm

    x_list=[]
    os.makedirs('../logs', exist_ok=True)
    for h_idx in tqdm.tqdm(range(1,6)):
        for w_idx in range(1,6):
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