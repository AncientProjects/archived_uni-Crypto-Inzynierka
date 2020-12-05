from base.base_model import BaseModel
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

keras = tf.keras


class ExperimentalModel(BaseModel):
    def __init__(self, train_data, config):
        super(ExperimentalModel, self).__init__(train_data, config)
        self.window_size = self.config.data_loader.window_size
        self.learning_rate = self.config.model.learning_rate
        self.momentum = self.config.model.momentum
        self.sequence_size = self.config.data_loader.sequences
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(10, input_shape=(self.window_size, 1)))
        self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(tf.keras.layers.Dense(self.sequence_size))

        self.model.compile(
            loss='mse',
            optimizer=self.config.model.optimizer,
            # optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.momentum, epsilon=1e-10),
            # optimizer=keras.optimizers.SGD(self.learning_rate, self.momentum),
            metrics=['mae'],
        )
        print(self.model.summary())

    def load_model(self):
        self.model = load_model(
            filepath=self.config.model.load_model_path,
            custom_objects=None, compile=True)
