import numpy as np

class Vectorizer:
    """ Vectorize words by each chars. Return numpy array."""

    def __init__(self, charFilePath, labelFilePath):
        self._charTable = dict()
        self._labelTable = dict()
        chars = open(charFilePath, "r", encoding="utf-8").read()
        labels = open(labelFilePath, "r", encoding="utf-8").read().strip().split("\n")
        for i, voc in enumerate(chars):
            self._charTable[voc] = i
        for i, term in enumerate(labels):
            self._labelTable[term] = i


    def vectorize(self, text, vec_width):
        "Matrix m by n = chars space in charTable by number of text chars. If texts are not enough, add padding."
        vec_height = len(self._charTable)
        vec = np.zeros((vec_height, vec_width))
        n = 0
        for i, char in enumerate(text):
            if n == vec_width:
                break
            if char in self._charTable:
                m = self._charTable[char]
                vec[m, n] = 1
                n += 1

        return vec


    def vectorizeLabel(self, labels):
        "Vectorize label to 1D vector."
        vec_height = len(self._labelTable)
        vec = np.zeros(vec_height)
        for label in labels:
            vec[self._labelTable[label]] = 1

        return vec
