from random import random

MAX_PARAM = 1.5
MIN_PARAM = 0.5


def generate_image_parameter(param_name_list: list) -> dict:
    return dict(zip(param_name_list,
                    [(MAX_PARAM - MIN_PARAM)*random()+MIN_PARAM
                     for _ in range(len(param_name_list))]))


def generate_image_parameter_list(param_name_list: list, n: int) -> list:
    return [generate_image_parameter(param_name_list) for _ in range(n)]


if __name__ == "__main__":
    print(generate_image_parameter_list(['a', 'b'], 3))
