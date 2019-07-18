from TrainDataGenerator.TFRecordsMaker.base_maker \
    import BaseMaker, SwitchableWriter

import numpy as np
import tensorflow as tf


class RegressionMaker(BaseMaker):
    def __init__(self, writer: SwitchableWriter):
        super(RegressionMaker, self).__init__(writer)

    def write(self, image_array: np.array, score: float):
        features = \
            tf.train.Features(
                feature={
                    'score':
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[score])),
                    'image':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[image_array.tobytes()])),
                }
            )

        self._write(features)
