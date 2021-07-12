import tensorflow as tf
from tensorflow.keras.layers import Layer

# https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137


class Attention(Layer):

    def __init__(self, return_sequences=False, **kwargs):
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = {"return_sequences": self.return_sequences}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
