import tensorflow
from tensorflow.keras.models import load_model

from base.base_model import BaseModel

tf = tensorflow
keras = tf.keras


class LSTMMultiOutputModel(BaseModel):
    def __init__(self, config):
        super(LSTMMultiOutputModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(100, input_shape=(self.window_size, 1)))
        self.model.add(tf.keras.layers.Dropout(0.1))
        # self.model.add(tf.keras.layers.LSTM(10, input_shape=(self.window_size, 1)))
        self.model.add(tf.keras.layers.Dense(self.sequence_size))

        self.model.compile(
            loss='mse',
            optimizer=self.config.model.optimizer,
            metrics=['mae'],
        )
        print(self.model.summary())

    def load_model(self):
        self.model = load_model(
            filepath=self.config.model.load_model_path,
            custom_objects=None, compile=True)

# optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.momentum, epsilon=1e-10),
# optimizer=keras.optimizers.SGD(self.learning_rate, self.momentum),
