import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import glob
import pandas as pd

from collections import OrderedDict

from misc.flag_holder import FlagHolder
from misc.logger import Logger
from misc.plot import plot

# options
@click.command()
# target
@click.option('-t', '--target_dir', type=str, required=True)
@click.option('-x', type=str, required=True)
@click.option('-y', type=str, default='')
@click.option('-h', '--hue', type=str, default=None, help='name of grouping variable that will produce lines with different colors')
@click.option('-l', '--log_path', type=str, default='', help='path of log')
@click.option('-s', '--save', is_flag=True, default=False, help='save results')

def main(**kwargs):
    plot_patch_shuffle(**kwargs)

def plot_patch_shuffle(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)

    target_path = os.path.join(FLAGS.target_dir, '**/*.csv')
    csv_paths   = sorted(glob.glob(target_path, recursive=True), key=lambda x: os.path.basename(x))

    df = None
    for csv_path in csv_paths:
        new_df = pd.read_csv(csv_path)
        legend = os.path.basename(new_df['weight'][0]).rstrip('_model.pth')
        new_df['legend'] = legend

        if df is None:
            df = new_df
        else:
            df = pd.concat([df, new_df], axis=0)
      
    plot(dataframe=df, x='num_devide', y='accuracy', hue='legend', log_path=os.path.join(FLAGS.log_path, 'plot_patch_shuffle.png'), save=FLAGS.save)

if __name__ == '__main__':
    main()