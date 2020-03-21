import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random
import numpy as np
import torch
import torchvision

def shuffle(x:torch.tensor, shuffle_dim:int, num_devide:int) -> torch.tensor:
    """
    shuffle input tensor along specified dim.

    Args
    - x:            input tensor. its size should be (h,w), (c,h,w), (b,c,h,w).
    - shuffle_dim:  dimension which applies shuffle operation.
    - num_devide:   number of division. x is devided into num_devide parts.
    
    Return: shuffled tensor.
    """
    shuffle_dim = shuffle_dim if shuffle_dim>=0 else int(len(x.size())+shuffle_dim) # make positive number

    assert 2 <= len(x.size())  <= 4
    assert x.size(shuffle_dim) > 0
    assert 0 <= shuffle_dim <  len(x.size())
    assert 1 <= num_devide  <= x.size(shuffle_dim)

    if num_devide == 1: return x

    end_indices = get_end_indices(x.size(shuffle_dim), num_devide)

    # devide x into num_devide parts and append them to the list
    devidied_tensor_parts = []
    begin_range = 0
    for end_range in end_indices:
        indices = torch.tensor(range(begin_range, end_range)).long()
        
        devidied_tensor_part = torch.index_select(x, dim=shuffle_dim, index=indices)
        devidied_tensor_parts.append(devidied_tensor_part)
        
        begin_range = end_range

    devidied_tensor_parts = random.sample(devidied_tensor_parts, len(devidied_tensor_parts))

    return torch.cat(devidied_tensor_parts, dim=shuffle_dim)


def get_end_indices(size:int, num_devide:int)->list:
    """
    try to equally devide input into num_devide.
    """
    assert 1 <= num_devide <= size

    parts_sizes = sorted([(size + i) // num_devide for i in range(num_devide)], reverse=True)

    end_indices = []
    #end_indices.append(parts_sizes[0])
    for i in range(len(parts_sizes)):
        end_indices.append(sum(parts_sizes[:i+1]))
    
    return end_indices




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
        for j in range(1, 10):
            x_h = shuffle(x, shuffle_dim=2, num_devide=j)
            x_h_list.append(x_h[1])

            x_w = shuffle(x, shuffle_dim=3, num_devide=j)
            x_w_list.append(x_w[1])

            x_both = shuffle(x, shuffle_dim=2, num_devide=j)
            x_both = shuffle(x_both, shuffle_dim=3, num_devide=j)
            x_both_list.append(x_both[1])

        break

    torchvision.utils.save_image(torch.stack(x_h_list, dim=0), "../logs/horizontal_shuffle.png", padding=1)
    torchvision.utils.save_image(torch.stack(x_w_list, dim=0), "../logs/vertical_shuffle.png", padding=1)
    torchvision.utils.save_image(torch.stack(x_both_list, dim=0), "../logs/both_shuffle.png", padding=1)
