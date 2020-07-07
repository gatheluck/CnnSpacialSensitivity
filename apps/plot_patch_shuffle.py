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
@click.option('-l', '--log_path', type=str, default='', help='path of log')
@click.option('-s', '--save', is_flag=True, default=False, help='save results')
def main(**kwargs):
    plot_patch_shuffle(**kwargs)


def plot_patch_shuffle(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)

    target_path = os.path.join(FLAGS.target_dir, '**/*.csv')
    csv_paths = sorted(glob.glob(target_path, recursive=True))

    df = None
    for i, csv_path in enumerate(csv_paths):
        new_df = pd.read_csv(csv_path)
        # legend = os.path.basename(new_df['weight'][0]).rstrip('_model.pth')  # this operation is redandant for many users.

        # if there is no legend index in df, use index as legend.
        if 'legend' not in new_df.columns.to_list():
            new_df['legend'] = i

        if df is None:
            df = new_df
        else:
            df = pd.concat([df, new_df], axis=0)

    plot(dataframe=df, x='num_devide', y='accuracy', hue='legend', log_path=os.path.join(FLAGS.log_path, 'plot_patch_shuffle.png'), save=FLAGS.save)


if __name__ == '__main__':
    main()
