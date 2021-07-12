import tensorflow as tf
from tensorflow.keras.layers import Layer
from model_utils import Attention
from data import Data
import numpy as np


class Bothoven:
    def __init__(self, data):
        self.data = data
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=data.tokenizer.wc, output_dim=200, input_length=self.data.sentence_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            Attention(return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            Attention(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(data.tokenizer.wc, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    def summary(self):
        return self.model.summary()

    def train(self, num_epochs=1):
        checkpoint_filepath = 'checkpoints'
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            verbose=1)

        self.model.fit(self.data.dataset(), epochs=num_epochs, callbacks=[checkpoint_callback])

    def save(self):
        self.model.save("model.h5")

    def load(self):
        self.model = tf.keras.models.load_model("model.h5", custom_objects={'Attention': Attention})

    def generate_from_song(self, filename, length):
        seed = []
        notes = self.data.extract_notes_from_song(filename)
        for i in notes:
            if len(seed) == self.data.sentence_length:
                break
            seed.append(notes[i])
        return self.generate(length, seed)

    def generate(self, length, seed=None):
        seed = seed if seed is not None else np.random.randint(0, self.data.tokenizer.wc, self.data.sentence_length)
        original_length = len(seed)
        for i in range(length):
            probability = self.model.predict(seed[i:i + self.data.sentence_length])[0]
            note = np.random.choice(self.data.tokenizer.wc, 1, replace=False, p=probability)
            seed = np.append(seed, note)

        result = []
        for note in seed[original_length:]:
            result.append(self.data.tokenizer.index_to_word[str(note)])
        return result

    def generate_midi(self, length, seed=None):
        notes = self.generate(length, seed)
        # TODO
        return notes


if __name__ == "__main__":
    d = Data()
    d.load()
    # d.init(size=100)
    # d.save()
    bothoven = Bothoven(d)
