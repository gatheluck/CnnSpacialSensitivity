import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random
import numpy as np
import torch
import torchvision

def shuffle(x:torch.tensor, shuffle_dim:int, num_partition:int) -> torch.tensor:
    """
    shuffle input tensor along specified dim.

    Args
    - x:             input tensor. its size should be (h,w), (c,h,w), (b,c,h,w).
    - shuffle_dim:   dimension which applies shuffle operation.
    - num_partition: number of partition.
    
    Return: shuffled tensor.
    """
    assert 2 <= len(x.size()) < 5
    assert 0 <= shuffle_dim    < len(x.size())
    assert 0 <= num_partition < min(x.size(-1), x.size(-2)) 

    if num_partition == 0: return x

    shuffle_dim_size = x.size(shuffle_dim) 
    patch_size = int(np.ceil(shuffle_dim_size / (num_partition+1)))

    # devide x into (num_partition+1) parts and append them to the list
    devidied_tensor_parts = []
    for i in range(num_partition+1):
        is_last = (i==num_partition)
        end_range = x.size(shuffle_dim) if is_last else (i+1)*patch_size

        indices = torch.tensor(range(i*patch_size, end_range))
        
        devidied_tensor_part = torch.index_select(x, dim=shuffle_dim, index=indices)
        devidied_tensor_parts.append(devidied_tensor_part)

    devidied_tensor_parts = random.sample(devidied_tensor_parts, len(devidied_tensor_parts))

    return torch.cat(devidied_tensor_parts, dim=shuffle_dim)
    
if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
    dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, transform=transform, download=True)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    os.makedirs('../logs', exist_ok=True)

    for i, (x,t) in enumerate(loader):

        x_h_list=[]
        x_w_list=[]
        x_both_list=[]
        for j in range(6):
            x_h = shuffle(x, shuffle_dim=2, num_partition=j)
            x_h_list.append(x_h[1])

            x_w = shuffle(x, shuffle_dim=3, num_partition=j)
            x_w_list.append(x_w[1])

            x_both = shuffle(x, shuffle_dim=2, num_partition=j)
            x_both = shuffle(x_both, shuffle_dim=3, num_partition=j)
            x_both_list.append(x_both[1])

        break

    torchvision.utils.save_image(torch.stack(x_h_list, dim=0), "../logs/horizontal_shuffle.png", padding=1)
    torchvision.utils.save_image(torch.stack(x_w_list, dim=0), "../logs/vertical_shuffle.png", padding=1)
    torchvision.utils.save_image(torch.stack(x_both_list, dim=0), "../logs/both_shuffle.png", padding=1)
