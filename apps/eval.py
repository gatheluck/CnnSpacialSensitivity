import collections
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import torch
import torchvision
import tqdm

from misc.data import DatasetBuilder
from misc.flag_holder import FlagHolder
from misc.io import load_model
from misc.logger import Logger
from misc.metric import get_num_correct
from misc.model import ModelBuilder
from misc.plot import plot
from spatial_sensitivity.patch_shuffle import PatchShuffle
from spatial_sensitivity.patch_shuffle import eval_patch_shuffle


# options
@click.command()
# model
@click.option('-a', '--arch', type=str, required=True)
# model
@click.option('-w', '--weight', type=str, required=True, help='model weight path')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=1024)
@click.option('--num_samples', type=int, default=-1)
# patch shuffle
@click.option('--max_num_devide', type=int, default=10)
@click.option('-k', '--top_k', type=int, default=1)
# log
@click.option('-l', '--log_dir', type=str, required=True)
@click.option('-s', '--suffix', type=str, default='')
def main(**kwargs):
    eval(**kwargs)


def eval(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    FLAGS.dump(path=os.path.join(FLAGS.log_dir, 'flags{}.json'.format(FLAGS.suffix)))

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)

    # model (load from checkpoint)
    num_classes = dataset_builder.num_classes
    model = ModelBuilder(num_classes=num_classes, pretrained=False)[FLAGS.arch].cuda()
    load_model(model, FLAGS.weight)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    eval_patch_shuffle(model, dataset_builder, **FLAGS._dict)


if __name__ == '__main__':
    main()
