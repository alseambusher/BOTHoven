import glob
import random
import pretty_midi as pm
import numpy as np
import json
import config
from tokenizer import Tokenizer, PAD
import tensorflow as tf
import sys
import logging

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Data:
    def __init__(self, batch_size=15):
        self.songs = []
        self.tokenizer = Tokenizer()
        self._loaded = False
        self.batch_size = batch_size
        self.sentence_length = 50


    @staticmethod
    def _load_file(filename, fps):
        midi = pm.PrettyMIDI(filename)
        return midi.instruments[0].get_piano_roll(fs=fps)  # numpy array (128 notes, total_time/fps) is the shape


    @staticmethod
    def _load_midi_generator(size, fps):
        _files = glob.glob(config.DATASET_FOLDER + "/**/*.midi")
        for _file in random.sample(_files, size if size > 0 else len(_files)):
            yield Data._load_file(_file, fps)

    def extract_notes_generator(self, size=-1, fps=5):
        song_number = 0
        for midi in Data._load_midi_generator(size, fps):
            logging.info(song_number)
            # Ignore where nothing is being played
            _filter = np.where(midi > 0)
            # Multiple notes can be played at a given time, so get unique to find times where something was played
            _filter_times = np.unique(_filter[1])

            all_notes = {}
            for time in _filter_times:
                all_notes[str(time)] = self.tokenizer.add_word(_filter[0][np.where(_filter[1] == time)])

            logging.info(song_number)
            song_number += 1
            yield all_notes

    # For inference
    def extract_notes_from_song(self, filename, fps=5):
        _filter = np.where(Data._load_file(filename, fps) > 0)
        _filter_times = np.unique(_filter[1])

        all_notes = {}
        for time in _filter_times:
            try:
                all_notes[str(time)] = self.tokenizer.word_to_index[
                    self.tokenizer.note_to_word(_filter[0][np.where(_filter[1] == time)])
                ]
            except:
                logging.warning("skipping")

        return all_notes

    def sentence_generator(self):
        songs = self.songs if self._loaded else self.extract_notes_generator()
        for song in songs:
            timestamps = [int(i) for i in song.keys()]
            X = [self.tokenizer.word_to_index[PAD]]*self.sentence_length
            for _time in range(int(timestamps[0]), int(timestamps[-1] + 1)):
                X.pop(0)
                if _time in timestamps:
                    X.append(song[str(_time)])
                else:
                    X.append(self.tokenizer.word_to_index[PAD])

                if _time + 1 in timestamps:
                    Y = song[str(_time + 1)]
                else:
                    Y = self.tokenizer.word_to_index[PAD]
                yield X[:], Y

    def dataset(self):
        dataset = tf.data.Dataset.from_generator(self.sentence_generator, (tf.int32, tf.int32))
        dataset = dataset.batch(self.batch_size)
        # TODO shuffle
        return dataset

    def init(self, size=1):
        self.songs = [x for x in self.extract_notes_generator(size=size)]
        self._loaded = True

    def save(self):
        with open("out.json", "w") as f:
            json.dump(self.songs, f)
        self.tokenizer.save()

    def load(self):
        with open("out.json", "r") as f:
            self.songs = json.load(f)
        self.tokenizer.load()
        self._loaded = True


if __name__ == "__main__":
    d = Data()
    d.init(size=100)
    d.save()
