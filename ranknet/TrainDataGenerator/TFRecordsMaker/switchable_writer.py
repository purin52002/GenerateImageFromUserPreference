import tensorflow as tf
from pathlib import Path
from .util import TRAIN, DATASET_TYPE_LIST, EXTENSION


class SwitchableWriter:
    def __init__(self, save_file_dir: str):
        self.writer_dict = \
            {key: tf.io.TFRecordWriter(
                str(Path(save_file_dir)/(key+EXTENSION)))
                for key in DATASET_TYPE_LIST}

        self.dataset_iter = iter(DATASET_TYPE_LIST)
        self.switcher = self.dataset_iter.__next__()

    def write(self, record: str):
        self.writer_dict[self.switcher].write(record)


class AutoSwitchableWriter(SwitchableWriter):
    def __init__(self, save_file_dir: str, rate_dict: dict, data_length: int):
        super(AutoSwitchableWriter, self).__init__(save_file_dir)

        self.write_count_dict = {key: 0 for key in DATASET_TYPE_LIST}
        self.data_length_dict = \
            {key: int(data_length * rate_dict[key])
                for key in DATASET_TYPE_LIST}

    def write(self, record: str):
        super(AutoSwitchableWriter, self).write(record)

        self.write_count_dict[self.switcher] += 1

        write_count = self.write_count_dict[self.switcher]
        data_length = self.data_length_dict[self.switcher]

        if write_count > data_length:
            try:
                self.switcher = self.dataset_iter.__next__()
            except StopIteration:
                self.switcher = TRAIN
