from scipy.stats import multivariate_normal
import numpy as np
from image_parameter_generator import MAX_PARAM, MIN_PARAM
import math
GRID_NUM = 128


def _get_key_list_without_score(data_dict: dict):
    return [
        key for key in data_dict.keys() if key != 'score']


def _make_gauss_data_dict(data_dict: dict):
    return {
        'mu': [data_dict[key] for key in _get_key_list_without_score(data_dict)],
        'amp': math.log(data_dict['score'])
    }


def make_gauss_graph_variable(data_dict_list: list):
    x = np.linspace(MIN_PARAM, MAX_PARAM, GRID_NUM)
    y = np.linspace(MIN_PARAM, MAX_PARAM, GRID_NUM)

    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    sharpness = 1000
    sigma = np.array([[1/sharpness, 0], [0, 1/sharpness]])
    gauss_data_dict_list = [_make_gauss_data_dict(
        data_dict) for data_dict in data_dict_list]

    Z = \
        sum(
            [gauss_data_dict['amp'] *
                multivariate_normal(gauss_data_dict['mu'], sigma).pdf(pos)
                for gauss_data_dict in gauss_data_dict_list])

    return X, Y, Z, _get_key_list_without_score(data_dict_list[0])+['score']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # grid_num = 128
    # x = np.linspace(-6, 6, grid_num)
    # y = np.linspace(-6, 6, grid_num)

    # X, Y = np.meshgrid(x, y)
    # pos = np.dstack((X, Y))

    # mu = np.array([0.5, 6])
    # sigma = np.array([[1, 0.2], [-0.2, 1]])
    # Z1 = 2*

    # mu = np.array([-0.5, -1])
    # sigma = np.array([[1, 0.2], [-0.2, 1]])
    # Z2 = 1*multivariate_normal(mu, sigma).pdf(pos)

    ax = Axes3D(plt.figure())
    data_dict1 = {'a': 0.8, 'b': 1.0, 'score': 1}
    data_dict2 = {'a': 1.2, 'b': 0.6, 'score': 8}
    X, Y, Z, _ = make_gauss_graph_variable([data_dict1])
    ax.plot_surface(X, Y, Z, cmap='coolwarm', cstride=1, rstride=1)
    plt.show()
