from base.base_model import BaseModel
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
keras = tf.keras


class LSTMTestModel(BaseModel):
    def __init__(self, train_data, config):
        super(LSTMTestModel, self).__init__(train_data, config)
        self.window_size = self.config.data_loader.window_size
        self.learning_rate = self.config.model.learning_rate
        self.momentum = self.config.model.momentum
        self.sequence_size = self.config.data_loader.sequences
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(self.window_size, 1)))
        self.model.add(LSTM(100, input_shape=(self.window_size, 1)))
        self.model.add(Dense(self.sequence_size))

        self.model.compile(
            loss='mse',
            # optimizer=self.config.model.optimizer,
            optimizer=keras.optimizers.SGD(self.learning_rate, self.momentum),
            metrics=['mae'],
        )

    def load_model(self):
        self.model = load_model(
            filepath=self.config.model.load_model_path,
            custom_objects=None, compile=True)
