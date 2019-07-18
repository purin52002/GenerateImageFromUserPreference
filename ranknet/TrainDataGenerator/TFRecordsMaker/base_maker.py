from TrainDataGenerator.TFRecordsMaker.switchable_writer \
    import SwitchableWriter
import tensorflow as tf


class BaseMaker:
    def __init__(self, writer: SwitchableWriter):
        self.writer = writer

    def _write(self, features: tf.train.Features):
        example = tf.train.Example(features=features)
        record = example.SerializeToString()
        self.writer.write(record)
