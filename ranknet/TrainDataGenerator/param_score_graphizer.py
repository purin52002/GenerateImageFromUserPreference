import matplotlib.pyplot as plt
import numpy as np
from gauss import make_gauss_graph_variable
from mpl_toolkits.mplot3d import Axes3D

from argparse import ArgumentParser
import csv


def scatter_graph(param_file_path_list: list, figure: plt.Figure):
    color_list = ['red', 'blue', 'green', 'yellow']

    for index, path in enumerate(param_file_path_list):
        with open(path) as param_file:
            read_dict = \
                csv.DictReader(param_file, delimiter=",", quotechar='"')

            key_list = read_dict.fieldnames

            dict_list = \
                [dict(zip(row.keys(), map(float, row.values())))
                for row in read_dict]

        score_list = [item['score'] for item in dict_list]

        for key in key_list:
            if key == 'score':
                continue

            item_list = [item[key] for item in dict_list]

            if key not in ax_dict:
                ax_dict[key] = figure.add_subplot(2, 2, key_list.index(key)+1)
                ax_dict[key].set_title(key)

            ax_dict[key].scatter(
                np.array(item_list), np.array(score_list),
                c=color_list[index], label='実験%i' % (index+1))

    plt.show()


def gauss_graph(param_file_list: list, figure: plt.Figure):
    for param_index, param_file in enumerate(param_file_list):
        read_dict = csv.DictReader(param_file, delimiter=",", quotechar='"')
        key_list = read_dict.fieldnames
        assert len(key_list) == 3

        dict_list = [dict(zip(row.keys(), map(float, row.values())))
                     for row in read_dict]

        X, Y, Z, label_list = make_gauss_graph_variable(dict_list)

        ax = Axes3D(figure)
        ax.set_xlabel(label_list[0], size=16)
        ax.set_ylabel(label_list[1], size=16)
        ax.set_zlabel(label_list[2], size=16)
        ax.plot_surface(X, Y, Z, cmap='coolwarm', cstride=1, rstride=1)

    plt.show()


SCATTER = 'scatter'
GAUSS = 'gauss'
GRAPH_TYPE_LIST = [SCATTER, GAUSS]


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-p', '--param_paths', nargs='*', required=True)
    parser.add_argument('-g', '--graph_type',
                        choices=GRAPH_TYPE_LIST, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    figure = plt.figure(figsize=(8, 8))
    ax_dict = {}

    if args.graph_type == SCATTER:
        scatter_graph(args.param_paths, figure)
    else:
        gauss_graph(args.param_paths, figure)
