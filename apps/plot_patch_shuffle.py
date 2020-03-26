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
@click.option('-l', '--log_path', type=str, default='', help='path of log')
@click.option('-s', '--save', is_flag=True, default=False, help='save results')

def main(**kwargs):
    plot_patch_shuffle(**kwargs)

def plot_patch_shuffle(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    # FLAGS.summary()

    target_path = os.path.join(FLAGS.target_dir, '**/*.csv')
    csv_paths   = sorted(glob.glob(target_path, recursive=True), key=lambda x: os.path.basename(x))

    print(csv_paths)

    df = None
    for csv_path in csv_paths:
        if df is None:
            df = pd.read_csv(csv_path)
        else:
            df = pd.concat([df, pd.read_csv(csv_path)], axis=0)
      
    print(df)
    plot(dataframe=df, x='num_devide', y='accuracy', log_path=os.path.join(FLAGS.log_path, 'plot_patch_shuffle.png'), save=FLAGS.save)

    #     log_dir = os.path.join(os.path.dirname(weight_path), 'plot')
    #     os.makedirs(log_dir, exist_ok=True)

    #     basename = os.path.basename(weight_path)
    #     basename, _ = os.path.splitext(basename) 
    #     log_path = os.path.join(log_dir, basename)+'.png'

    #     cmd = 'python plot.py \
    #         -t {target_dir} \
    #         -x {x} \
    #         -s \
    #         -l {log_path}'.format(
    #             target_dir=weight_path,
    #             x=FLAGS.x,
    #             log_path=log_path)

    #     # add y
    #     if FLAGS.y != '':
    #         cmd += ' -y {y}'.format(y=FLAGS.y)

    #     # add flag command
    #     if FLAGS.plot_all:
    #         cmd += ' --plot_all'

    #     subprocess.run(cmd.split(), cwd=run_dir)

if __name__ == '__main__':
    main()