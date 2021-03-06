import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from misc.flag_holder import FlagHolder


@click.command()
@click.option("-p", "--csv_path", type=str, required=True)
@click.option("-x", type=str, required=True)
@click.option("-y", type=str, default="")
@click.option(
    "-h",
    "--hue",
    type=str,
    default=None,
    help="name of grouping variable that will produce lines with different colors",
)
@click.option("-l", "--log_path", type=str, default="", help="path of log")
@click.option("-s", "--save", is_flag=True, default=False, help="save results")
def main(**kwargs):
    plot(**kwargs)


def plot(**kwargs):
    # parse keyword arugments
    required_kwargs = "x y hue log_path save".split()
    for required_kwarg in required_kwargs:
        if required_kwarg not in kwargs.keys():
            raise ValueError("invalid args")

    has_dataframe = "dataframe" in kwargs.keys()
    has_csv_path = "csv_path" in kwargs.keys()

    if (has_dataframe and has_csv_path) or (not has_dataframe and not has_csv_path):
        raise ValueError("invalid args")

    # set flag
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    if has_csv_path:
        df = pd.read_csv(FLAGS.csv_path)
    else:
        df = FLAGS.dataframe

    fig = plt.figure()

    if FLAGS.y == "":
        raise ValueError('please specify "y"')

    sns.lineplot(x=FLAGS.x, y=FLAGS.y, hue=FLAGS.hue, ci="sd", data=df)

    if FLAGS.save:
        plt.close()
        if FLAGS.log_path == "":
            raise ValueError('please specify "log_path"')
        os.makedirs(os.path.dirname(FLAGS.log_path), exist_ok=True)
        fig.savefig(FLAGS.log_path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
