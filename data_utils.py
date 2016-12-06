# -*- coding: utf-8 -*-
"""Data utility module.

Todo:
    * Iterator for sentence independent data

"""
import os
import warnings
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

    def word_set(self):
        return set(self._w2i.keys())

    def iarr2warr(self, iarr):
        w = []
        for ir in iarr:
            iw = []
            w.append(iw)
            for i in ir:
                iw.append(self.i2w(i))
        return w

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

    @staticmethod
    def vocab_index_map(vocab_a, vocab_b):
        a2b = {}
        b2a = {}
        for w in vocab_a.word_set():
            a2b[vocab_a.w2i(w)] = vocab_b.w2i(w)
        for w in vocab_b.word_set():
            b2a[vocab_b.w2i(w)] = vocab_a.w2i(w)
        return a2b, b2a

    @staticmethod
    def list_ids_from_file(filepath, vocab):
        l = []
        with codecs.open(filepath, 'r', 'utf-8') as ifp:
            for line in ifp:
                word = line.strip().split()[0]
                l.append(vocab.w2i(word))
        return l

class DataIterator(object):
    def __init__(self, vocab=None, file_path=None):
        if vocab is not None:
            self._vocab = vocab
            self._padding_id = vocab.eos_id
            self._data, self._lidx, self._lkeys, self._max_seq_len \
            = self._parse_file(file_path)


    def _parse_sentence(self, sentence):
        indexes = [self._vocab.w2i(word) for word in sentence.split()]
        # return [self._vocab.sos_id] + indexes + [self._vocab.eos_id]
        return indexes + [self._vocab.eos_id]

    def _parse_file(self, filepath):
        data = []
        label_idx = []
        label_keys = []
        max_seq_len = 0
        with open(filepath) as ifp:
            for line in ifp:
                doc = json.loads(line)
                label_idx.append(len(data))
                label_keys.append(doc['key'])
                seq_len = 0
                for s in doc['lines']:
                    line = self._parse_sentence(s)
                    seq_len += len(line)
                    data += line
                if seq_len > max_seq_len:
                    max_seq_len = seq_len
        data = np.array(data, np.int32)
        return data, label_idx, label_keys, max_seq_len

    def init_batch(self, batch_size, num_steps):
        if num_steps < 1:
            warnings.warn("num_steps has to be more than 0.")
        self._batch_size = batch_size
        self._num_steps = num_steps
        self.x = np.zeros([batch_size, num_steps], np.int32)
        self.y = np.zeros([batch_size, num_steps], np.int32)
        self.w = np.zeros([batch_size, num_steps], np.uint8)
        self.l = [[None for _ in range(num_steps)] for _ in range(batch_size)]
        self.seq_len = np.zeros([batch_size], np.int32)
        self.seq_len[:] = self._num_steps
        self._pointers = [0] * batch_size
        distance = len(self._data) / batch_size
        for i in range(batch_size):
            self._pointers[i] = i * distance
        random.shuffle(self._pointers)
        # XXX: this will cut off the left-over data
        self._epoch_tokens = distance
        self._read_tokens = [0 for _ in range(batch_size)]

    def next_batch(self):
        if any((t + 1) >= self._epoch_tokens for t in self._read_tokens):
            return None, None, None, None, None
        # reset old data
        self.x[:], self.y[:], self.w[:] = self._padding_id, self._padding_id, 0
        for i in range(len(self.l)):
            for j in range(len(self.l[0])):
                self.l[i][j] = -1
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
        return self.x, self.y, self.w, self.l, self.seq_len

    def iterate_epoch(self, batch_size, num_steps=-1):
        self.init_batch(batch_size, num_steps)
        while True:
            x, y, w, l, seq_len = self.next_batch()
            if x is None:
                break
            yield x, y, w, l, seq_len

class DefIterator(DataIterator):
    # def _parse_sentence(self, sentence):
    #     indexes = [self._vocab.w2i(word) for word in sentence.split()]
    #     return [self._vocab.sos_id] + indexes + [self._vocab.eos_id]

    def _parse_file(self, filepath):
        data = []
        label_idx = []
        label_keys = []
        max_seq_len = 0
        with open(filepath) as ifp:
            for line in ifp:
                doc = json.loads(line)
                label_idx.append(len(data))
                label_keys.append(self._vocab.w2i(doc['key']))
                seq_len = 0
                for s in doc['lines']:
                    line = self._parse_sentence(s)
                    seq_len += len(line)
                    data += line
                if seq_len > max_seq_len:
                    max_seq_len = seq_len
        padded_data = np.zeros([max_seq_len * len(label_keys)], np.int32)
        padded_data[:] = self._padding_id
        padded_label_idx = []
        prev_idx = 0
        for i in range(1, len(label_idx)):
            padded_label_idx.append((i-1)*max_seq_len)
            cur_seq_start = (i-1)*max_seq_len
            cur_seq_end = cur_seq_start + label_idx[i] - prev_idx
            padded_data[cur_seq_start:cur_seq_end] = data[prev_idx:label_idx[i]]
            prev_idx = label_idx[i]
        # data = np.array(data, np.int32)
        return padded_data, padded_label_idx, label_keys, max_seq_len

    def _shuffle_data(self):
        shuff_keys = []
        shuff_data = np.zeros(self._data.shape, np.int32)
        perm_index = range(len(self._lidx))
        random.shuffle(perm_index)
        for i, j in enumerate(perm_index):
            shuff_keys.append(self._lkeys[j])
            istart = i * self._max_seq_len
            iend = (i + 1) * self._max_seq_len
            jstart = j * self._max_seq_len
            jend = (j + 1) * self._max_seq_len
            shuff_data[istart:iend] = self._data[jstart:jend]
        self._data = shuff_data
        self._lkeys = shuff_keys

    def init_batch(self, batch_size, num_steps=-1):
        if num_steps != -1 and num_steps != self._max_seq_len:
            warnings.warn("num_steps is not the same as max sequence len!")
        if num_steps == -1:
            num_steps = self._max_seq_len
        super(DefIterator,self).init_batch(batch_size, num_steps)
        #XXX: this will cut off some definition
        distance = len(self._lidx) / batch_size
        for i in range(batch_size):
            self._pointers[i] = self._lidx[i * distance]
        random.shuffle(self._pointers)
        self._shuffle_data()
        self.l = np.zeros([batch_size, num_steps], np.int32)

    def next_batch(self):
        x, _, _, _, _ = super(DefIterator, self).next_batch()
        if x is None:
            return None, None, None, None, None
        self.y[:, -1] = self._padding_id
        self.w[:] = 0
        self.seq_len = np.sum(self.y!=self._padding_id, axis=1) + 1
        for i in range(self._batch_size):
            self.w[i, 1:self.seq_len[i]] = 1
        return self.x, self.y, self.w, self.l, self.seq_len

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

def serialize_corpus(data_dir, split=['train', 'valid', 'test']):
    for s in split:
        corpus2bow(os.path.join(data_dir, '{}.jsonl'.format(s)),
                   os.path.join(data_dir, 'bow_vocab.txt'),
                   os.path.join(data_dir, '{}_bow.pickle'.format(s)))
        serialize_iterator(os.path.join(data_dir, '{}.jsonl'.format(s)),
                   os.path.join(data_dir, 'vocab.txt'),
                   os.path.join(data_dir, '{}_iter.pickle'.format(s)))
