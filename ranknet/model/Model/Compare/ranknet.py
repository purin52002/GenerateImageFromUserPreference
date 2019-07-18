import tensorflow as tf
from UserPreferencePredictor.Model.Compare.dataset \
    import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL
from UserPreferencePredictor.Model.Compare.evaluate_network \
    import EvaluateNetwork
from pathlib import Path
import numpy as np
from PIL.Image import Image

layers = tf.keras.layers
losses = tf.keras.losses


class RankNet:
    SCOPE = 'predict_model'
    PREDICTABLE_MODEL_FILE_NAME = 'predictable_model.h5'
    TRAINABLE_MODEL_FILE_NAME = 'trainable_model.h5'

    def __init__(self):
        with tf.name_scope(RankNet.SCOPE):
            evaluate_network = EvaluateNetwork()

            input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)

            left_input = tf.keras.Input(shape=input_shape)
            right_input = tf.keras.Input(shape=input_shape)

            left_output = evaluate_network(left_input)
            right_output = evaluate_network(right_input)

            concated_output = layers.Concatenate()([left_output, right_output])

            self.predictable_model = tf.keras.Model(inputs=left_input,
                                                    outputs=left_output)

            self.trainable_model = tf.keras.Model(inputs=[left_input,
                                                          right_input],
                                                  outputs=concated_output)

            loss = \
                tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)
            self.trainable_model.compile(optimizer='adam', loss=loss)

    def train(self, dataset: tf.data.Dataset, *, log_dir_path: str,
              valid_dataset: tf.data.Dataset, epochs=10, steps_per_epoch=30):
        callbacks = tf.keras.callbacks

        cb = []

        # cb.append(callbacks.EarlyStopping())

        if log_dir_path is not None:
            cb.append(callbacks.TensorBoard(log_dir=log_dir_path,
                                            write_graph=True))

        self.trainable_model.fit(dataset, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=cb, validation_data=valid_dataset,
                                 validation_steps=10)

    def save(self, save_dir_path: str):
        save_dir_path = Path(save_dir_path)
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True)

        self.predictable_model.save_weights(
            str(Path(save_dir_path) /
                RankNet.PREDICTABLE_MODEL_FILE_NAME)
        )

        self.trainable_model.save_weights(
            str(Path(save_dir_path) /
                RankNet.TRAINABLE_MODEL_FILE_NAME))

    def load(self, load_dir_path: str):
        self.predictable_model.load_weights(
            str(Path(load_dir_path)/RankNet.PREDICTABLE_MODEL_FILE_NAME))

        self.trainable_model.load_weights(
            str(Path(load_dir_path)/RankNet.TRAINABLE_MODEL_FILE_NAME))

    def save_model_structure(self, save_dir_path: str):
        save_dir_path = Path(save_dir_path)
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True)

        tf.keras.utils.plot_model(self.predictable_model,
                                  str(save_dir_path/'predictable_model.png'),
                                  show_shapes=True)

        tf.keras.utils.plot_model(self.trainable_model,
                                  str(save_dir_path/'trainable_model.png'),
                                  show_shapes=True)

    def _image_to_array(self, image: Image):
        resized_image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        return np.asarray(resized_image).astype(np.float32)/255

    def predict(self, data_list: list):
        image_array_list = np.array([self._image_to_array(data['image'])
                                     for data in data_list])

        return self.predictable_model.predict(image_array_list)


if __name__ == '__main__':
    image_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)
    model = RankNet()

    def train(model):
        def make_dataset_from_numpy(data_length: int):
            data_shape = (data_length, )+image_shape

            like_data = np.random.normal(0.2, 0.03, data_shape)
            unlike_data = np.random.normal(0.8, 0.03, data_shape)
            label = np.zeros(data_length)

            dataset = tf.data.Dataset.from_tensor_slices(
                ((like_data, unlike_data), label)).batch(32).repeat()

            return dataset

        train_data_length = 100
        train_dataset = make_dataset_from_numpy(train_data_length)

        valid_data_length = 50
        valid_dataset = make_dataset_from_numpy(valid_data_length)

        log_dir_path = Path(__file__).parent/'log'
        model.train(train_dataset, log_dir_path=str(log_dir_path),
                    valid_dataset=valid_dataset)

    def predict(model):
        good = model.predict(np.random.normal(0.2, 0.03, (1,)+image_shape))
        bad = model.predict(np.random.normal(0.8, 0.03, (1,)+image_shape))

        print(f'good: {good}')
        print(f'bad: {bad}')

    def save(model):
        save_dir_path = Path(__file__).parent/'save'
        model.save(save_dir_path)

    def load(model):
        load_dir_path = Path(__file__).parent/'save'
        model.load(load_dir_path)

    train(model)
    save(model)

    new_model = RankNet()
    load(new_model)
    predict(new_model)
