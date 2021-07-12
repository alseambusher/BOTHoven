import json

PAD = "<PAD>"


class Tokenizer:

    def __init__(self):
        self.dictionary = {}
        self.index_to_word = {}
        self.word_to_index = {}
        self.wc = 0
        self.add_word(PAD)

    def add_word(self, word):
        if not isinstance(word, str):
            word = self.note_to_word(word)
        if word not in self.word_to_index:
            self.index_to_word[self.wc] = word
            self.word_to_index[word] = self.wc
            self.wc += 1
        return self.word_to_index[word]

    def note_to_word(self, note):
        return " ".join(list(note.astype(str)))

    def load(self):
        with open("index.json", "r") as f:
            data = json.load(f)
            self.wc = data["wc"]
            self.index_to_word = data["i2w"]
            self.word_to_index = data["w2i"]

    def save(self):
        with open("index.json", "w") as f:
            json.dump({"wc": self.wc,
                       "i2w": self.index_to_word,
                       "w2i": self.word_to_index}, f)
