import tensorflow
from tensorflow.keras.models import load_model
from tensorflow_core.python.keras.utils.vis_utils import plot_model

from base.base_model import BaseModel

tf = tensorflow
keras = tf.keras


class LSTMSingleOutputModel(BaseModel):
    def __init__(self, config):
        super(LSTMSingleOutputModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(50, input_shape=(self.window_size, 1), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(100, input_shape=(self.window_size, 1)))
        self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(tf.keras.layers.Dense(self.sequence_size))
        plot_model(self.model, to_file='model1_2_plot.png', show_shapes=True, show_layer_names=True)

        self.model.compile(
            loss='mse',
            optimizer=self.config.model.optimizer,
            # optimizer=keras.optimizers.SGD(self.learning_rate, self.momentum),
            metrics=['mae'],
        )
        print(self.model.summary())

    def load_model(self):
        self.model = load_model(
            filepath=self.dataset_model(),
            custom_objects=None, compile=True)

    def dataset_model(self):
        if self.config.data_loader.dataset == '1':
            return self.config.model.load_model_path
        return self.config.model.load_model_path_2d

# optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.momentum, epsilon=1e-10),
# optimizer=keras.optimizers.SGD(self.learning_rate, self.momentum),
