import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from misc.flag_holder import FlagHolder

@click.command()
@click.option('-c', '--csv_path', type=str, required=True)
@click.option('-x', type=str, required=True)
@click.option('-y', type=str, default='')
@click.option('-l', '--log_path', type=str, default='', help='path of log')
@click.option('-s', '--save', is_flag=True, default=False, help='save results')

def main(**kwargs):
    plot(**kwargs)

def plot(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    df = pd.read_csv(FLAGS.csv_path)

    fig = plt.figure()
    ax = fig.subplots()
    if FLAGS.y == '': raise ValueError('please specify "y"')
    sns.lineplot(x=FLAGS.x, y=FLAGS.y, ci="sd", data=df)

    if FLAGS.save:
        plt.close()
        if FLAGS.log_path == '': 
            raise ValueError('please specify "log_path"')
        os.makedirs(os.path.dirname(FLAGS.log_path), exist_ok=True)
        fig.savefig(FLAGS.log_path)
    else:
        plt.show()

if __name__ == '__main__':
    main()