# -*- coding: utf-8 -*-
"""Data utility module.

Todo:
    * Test case

"""
import os
import codecs
import numpy as np
import json
import cPickle
import random
from bisect import bisect_right

class Vocabulary(object):
    """
    A mapping between words and indexes. The code is adapted from
    `Rafal Jozefowicz's lm <https://github.com/rafaljozefowicz/lm>`_
    """
    def __init__(self):
        self._w2i = {}
        self._i2w = []
        self._i2freq = {}
        self._vocab_size = 0
        self._sos_id = None
        self._eos_id = None
        self._unk_id = None

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def sos_id(self):
        return self._sos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def sos(self):
        return "<s>"

    @property
    def eos(self):
        return "</s>"

    @property
    def unk(self):
        return "<unk>"

    def add(self, word, count):
        self._w2i[word] = self._vocab_size
        self._i2w.append(word)
        self._i2freq[self._vocab_size] = count
        self._vocab_size += 1

    def w2i(self, word):
        return self._w2i.get(word, self.unk_id)

    def i2w(self, index):
        return self._i2w[index]

    def finalize(self):
        self._sos_id = self.w2i(self.sos)
        self._eos_id = self.w2i(self.eos)
        self._unk_id = self.w2i(self.unk)

    def dense_bow(self, tokens, bow=None, lowcase=True):
        if bow is None:
            bow = np.zeros(self.vocab_size)
        for t in tokens:
            if lowcase:
                t = t.lower()
            index = self.w2i(t)
            if index == self.unk_id:
                continue
            bow[index] += 1
        return bow

    @staticmethod
    def from_vocab_file(filepath):
        vocab = Vocabulary()
        with codecs.open(filepath, 'r', 'utf-8') as ifp:
            for line in ifp:
                word, count = line.strip().split()
                vocab.add(word, int(count))
        vocab.finalize()
        return vocab

class DataIterator(object):
    def __init__(self, vocab=None, file_path=None):
        if vocab is not None:
            self._vocab = vocab
            self._data, self._lidx, self._lkeys = self._parse_file(file_path)
            self._padding_id = vocab.eos_id

    def _parse_sentence(self, sentence):
        indexes = [self._vocab.w2i(word) for word in sentence.split()]
        return [self._vocab.sos_id] + indexes + [self._vocab.eos_id]

    def _parse_file(self, filepath):
        data = []
        label_idx = []
        label_keys = []
        with open(filepath) as ifp:
            for line in ifp:
                doc = json.loads(line)
                label_idx.append(len(data))
                label_keys.append(doc['key'])
                for s in doc['lines']:
                    data += self._parse_sentence(s)
        data = np.array(data, np.int32)
        return data, label_idx, label_keys

    def init_batch(self, batch_size, num_steps):
        self._batch_size = batch_size
        self._num_steps = num_steps
        self.x = np.zeros([batch_size, num_steps], np.int32)
        self.y = np.zeros([batch_size, num_steps], np.int32)
        self.w = np.zeros([batch_size, num_steps], np.uint8)
        self.l = [[None for _ in range(num_steps)] for _ in range(batch_size)]
        self._pointers = [0] * batch_size
        distance = len(self._data) / batch_size
        for i in range(batch_size):
            self._pointers[i] = i * distance
        random.shuffle(self._pointers)
        # XXX: this will cut off the left-over data
        self._epoch_tokens = distance
        self._read_tokens = [0 for _ in range(batch_size)]

    def next_batch(self):
        if any(t > self._epoch_tokens for t in self._read_tokens):
            return None, None, None, None
        # reset old data
        self.x[:], self.y[:], self.w[:] = self._padding_id, self._padding_id, 0
        for i in range(len(self.l)):
            for j in range(len(self.l[0])):
                self.l[i][j] = None
        # populating new data
        for i, p in enumerate(self._pointers):
            num_tokens = self._num_steps
            if p + self._num_steps + 1 > len(self._data):
                num_tokens = len(self._data) - p - 1
            self.x[i, :num_tokens] = self._data[p: p + num_tokens]
            self.y[i, :num_tokens] = self._data[p + 1: p + num_tokens + 1]
            self.w[i, :num_tokens] = 1
            for j in range(num_tokens):
                self.l[i][j] = self._lkeys[bisect_right(self._lidx, p + j) - 1]
            # increment pointers
            self._pointers[i] = p + num_tokens
            self._read_tokens[i] += num_tokens
        return self.x, self.y, self.w, self.l

    def iterate_epoch(self, batch_size, num_steps):
        self.init_batch(batch_size, num_steps)
        while True:
            x, y, w, l = self.next_batch()
            if x is None:
                break
            yield x, y, w, l

def serialize_iterator(data_filepath, vocab_filepath, out_filepath):
    vocab = Vocabulary.from_vocab_file(vocab_filepath)
    loader = DataIterator(vocab=vocab, file_path=data_filepath)
    with open(out_filepath, 'w') as ofp:
        cPickle.dump(loader, ofp)

def corpus2bow(data_filepath, vocab_filepath, out_filepath):
    vocab = Vocabulary.from_vocab_file(vocab_filepath)
    corpus_bow = {}
    with open(data_filepath) as ifp:
        for i, line in enumerate(ifp):
            entry = json.loads(line)
            bow = np.zeros(vocab.vocab_size)
            for l in entry['lines']:
                bow = vocab.dense_bow(l.split(), bow, lowcase=True)
            corpus_bow[entry['key']] = bow
    with open(out_filepath, 'w') as ofp:
        cPickle.dump(corpus_bow, ofp)

def serialize_corpus(data_dir):
    split = ['train', 'valid', 'test']
    for s in split:
        corpus2bow(os.path.join(data_dir, '{}.jsonl'.format(s)),
                   os.path.join(data_dir, 'bow_vocab.txt'),
                   os.path.join(data_dir, '{}_bow.pickle'.format(s)))
        serialize_iterator(os.path.join(data_dir, '{}.jsonl'.format(s)),
                   os.path.join(data_dir, 'vocab.txt'),
                   os.path.join(data_dir, '{}_iter.pickle'.format(s)))
