from pathlib import Path
from exception import TFRecordsMakerException


class CombinationException(TFRecordsMakerException):
    pass


def _parse_scored_info(left_info: dict, right_info: dict):
    if not all([left == right
                for left, right in zip(left_info.keys(), right_info.keys())]):
        raise CombinationException

    return {key: (left_info[key], right_info[key]) for key in left_info.keys()}


def get_combination_yield(scored_dir_path_list: list):
    scored_dir_length = len(scored_dir_path_list)

    for left_index in range(0, scored_dir_length-1):
        for right_index in range(left_index+1, scored_dir_length):
            left_info = scored_dir_path_list[left_index]
            right_info = scored_dir_path_list[right_index]
            parsed_info = _parse_scored_info(left_info, right_info)
            yield parsed_info


def get_data_length(scored_dir_path_list: list):
    data_length = 0

    for parsed_info in get_combination_yield(scored_dir_path_list):
        left_dir_path, right_dir_path = parsed_info['param']
        left_score, right_score = parsed_info['score']

        if left_score != right_score:
            left_image_num = len(list(Path(left_dir_path).iterdir()))
            right_image_num = len(list(Path(right_dir_path).iterdir()))
            data_length += left_image_num*right_image_num

    return data_length
