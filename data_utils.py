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

######################################################
# Vocabulary
######################################################

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
                parts = line.strip().split()
                count = 0
                word = parts[0]
                if len(parts) > 1:
                    count = int(parts[1])
                vocab.add(word, count)
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

    @staticmethod
    def create_vocab_mask(keep_vocab, full_vocab):
        mask = np.zeros(full_vocab.vocab_size)
        for w in keep_vocab.word_set():
            mask[full_vocab.w2i(w)] = 1
        return mask

######################################################
# DataIterator
######################################################

class DataIterator(object):
    """
    Iterate over text data

    kwargs:
        * x_vocab: Vocabulary for inputs (default: vocab)
        * y_vocab: Vocabulary for targets (default: vocab)
        * sos: Add start of sentence ID (default: False)
        * eos: Add end of sentence ID (default: True)
        * shuffle_data: Shuffle data between epoch (default: True)
    """
    def __init__(self, vocab=None, file_path=None, **kwargs):
        self._kwargs = kwargs
        self._x_vocab, self._y_vocab = None, None
        self._x_padding_id, self._y_padding_id = None, None
        self._add_sos, self._add_eos = False, True
        self._shuffle_data = True
        if 'x_vocab' in self._kwargs:
            self._x_vocab = self._kwargs['x_vocab']
            self._x_padding_id = self._x_vocab.eos_id
        if 'y_vocab' in self._kwargs:
            self._y_vocab = self._kwargs['y_vocab']
            self._y_padding_id = self._y_vocab.eos_id
        if 'sos' in self._kwargs:
            self._add_sos = self._kwargs['sos']
        if 'eos' in self._kwargs:
            self._add_eos = self._kwargs['eos']
        if 'shuffle_data' in self._kwargs:
            self._shuffle_data = self._kwargs['shuffle_data']
        if vocab is not None:
            self._vocab = vocab
            self._padding_id = vocab.eos_id
            if self._x_padding_id is None:
                self._x_padding_id = vocab.eos_id
            if self._y_padding_id is None:
                self._y_padding_id = vocab.eos_id
            self._parse_file(file_path)

    def _parse_sentence(self, sentence, vocab=None):
        if vocab is None:
            vocab = self._vocab
        indexes = [vocab.w2i(word) for word in sentence.split()]
        # return [self._vocab.sos_id] + indexes + [self._vocab.eos_id]
        if self._add_sos:
            indexes.insert(0, vocab.sos_id)
        if self._add_eos:
            indexes.append(vocab.eos_id)
        return indexes

    def _append_data_if_vocab(self, doc, vocab, data):
        if vocab is None:
            return 0, 0
        seq_len = 0
        num_lines = 0
        for s in doc['lines']:
            sen = self._parse_sentence(s, vocab)
            seq_len += len(sen)
            num_lines += 1
            data.extend(sen)
        return seq_len, num_lines

    def _parse_file(self, filepath):
        data, data_x, data_y, label_idx, label_keys  = [], [], [], [], []
        max_seq_len = 0
        num_lines = 0
        with open(filepath) as ifp:
            for line in ifp:
                doc = json.loads(line)
                label_idx.append(len(data))
                label_keys.append(doc['key'])
                seq_len, lines = self._append_data_if_vocab(
                    doc, self._vocab, data)
                _, __ = self._append_data_if_vocab(doc, self._x_vocab, data_x)
                _, __ = self._append_data_if_vocab(doc, self._y_vocab, data_y)
                if seq_len > max_seq_len:
                    max_seq_len = seq_len
                num_lines += lines
        data.insert(0, self._padding_id)
        self._data = np.array(data, np.int32)
        if len(data_x) == 0:
            self._data_x = self._data
            self._x_vocab = self._vocab
        else:
            data_x.insert(0, self._x_vocab.eos_id)
            self._data_x = np.array(data_x, np.int32)
        if len(data_y) == 0:
            self._data_y = self._data
            self._y_vocab = self._vocab
        else:
            data_y.insert(0, self._y_vocab.eos_id)
            self._data_y = np.array(data_y, np.int32)

        self._lidx = label_idx
        self._lkeys = label_keys
        self._max_seq_len = max_seq_len
        self._num_lines = num_lines

    def _find_label(self, data_pointer):
        return self._lkeys[bisect_right(self._lidx, data_pointer) - 1]

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
        self._pointers = np.zeros([batch_size], np.int32)
        distance = (len(self._data) - 1) / batch_size
        left_over = (len(self._data) - 1) % batch_size
        self._distances = np.zeros([batch_size], np.int32)
        self._distances[:] = distance
        self._distances[0:left_over] += 1
        cur_pos = 0
        for i in range(batch_size):
            self._pointers[i] = cur_pos
            cur_pos += self._distances[i]
        if self._shuffle_data:
            p = np.random.permutation(self._batch_size)
            self._pointers = self._pointers[p]
            self._distances = self._distances[p]
        self._read_tokens = np.zeros([batch_size], np.int32)

    def next_batch(self):
        if all(self._read_tokens >= self._distances):
            return None, None, None, None, None
        # reset old data
        self.x[:], self.y[:] = self._x_padding_id, self._y_padding_id
        self.w[:], self.seq_len[:] = 0, 0
        for i in range(len(self.l)):
            for j in range(len(self.l[0])):
                self.l[i][j] = -1
        # populating new data
        for i_batch in range(self._batch_size):
            for i_token in range(self._num_steps):
                if self._read_tokens[i_batch] >= self._distances[i_batch]:
                    continue
                cur_pos = self._pointers[i_batch]
                self.x[i_batch, i_token] = self._data_x[cur_pos]
                self.y[i_batch, i_token] = self._data_y[cur_pos + 1]
                self.w[i_batch, i_token] = 1
                self.seq_len[i_batch] += 1
                self._pointers[i_batch] += 1
                self._read_tokens[i_batch] += 1
        return self.x, self.y, self.w, self.l, self.seq_len

    def iterate_epoch(self, batch_size, num_steps=-1):
        self.init_batch(batch_size, num_steps)
        while True:
            x, y, w, l, seq_len = self.next_batch()
            if x is None:
                break
            yield x, y, w, l, seq_len

######################################################
# Setnence Iterator
######################################################

class SentenceIterator(DataIterator):
    """
    Iterate over sentence with padding, add is_new_sen() to check whether the
    current batch from a new set of sentences.

    kwargs:
        * sos: Add start of sentence ID (default: True)
    """

    def __init__(self, vocab=None, file_path=None, **kwargs):
        if 'sos' not in kwargs:
            kwargs['sos'] = True
        super(SentenceIterator, self).__init__(vocab, file_path, **kwargs)
        self._sen_idx = []
        eos_id = self._vocab.eos_id
        for i, wid in enumerate(self._data):
            if wid == eos_id and i + 1 < len(self._data) :
                self._sen_idx.append(i+1)

    def init_batch(self, batch_size, num_steps):
        if num_steps < 1:
            warnings.warn("num_steps has to be more than 0.")
        self._batch_sen_idx = list(self._sen_idx)
        if self._shuffle_data:
            random.shuffle(self._batch_sen_idx)
        self._batch_size = batch_size
        self._num_steps = num_steps
        self.x = np.zeros([batch_size, num_steps], np.int32)
        self.y = np.zeros([batch_size, num_steps], np.int32)
        self.w = np.zeros([batch_size, num_steps], np.uint8)
        self.l = [[None for _ in range(num_steps)] for _ in range(batch_size)]
        self.seq_len = np.zeros([batch_size], np.int32)
        self._pointers = np.zeros([batch_size], np.int32)
        distance = len(self._sen_idx) / batch_size
        left_over = len(self._sen_idx) % batch_size
        self._distances = np.array([distance for _ in range(batch_size)],
                                  dtype=np.int32)
        self._distances[0:left_over] += 1
        cur_pos = 0
        for i in range(batch_size):
            self._pointers[i] = cur_pos
            cur_pos += self._distances[i]
        self._read_sentences = np.array([0 for _ in range(batch_size)],
                                        dtype=np.int32)
        self._read_tokens = np.array([0 for _ in range(batch_size)],
                                     dtype=np.int32)
        self._new_sentence_set = True

    def next_batch(self):
        # increment sentence
        if self._read_tokens.sum() == -1 * self._batch_size:
            self._new_sentence_set = True
            self._pointers[:] += 1
            self._read_sentences[:] += 1
            self._read_tokens[:] = 0
            if all(self._read_sentences >= self._distances):
                return None, None, None, None, None
        elif self._read_tokens.sum() == 0:
            self._new_sentence_set = True
        else:
            self._new_sentence_set = False
        # reset old data
        self.x[:], self.y[:] = self._padding_id, self._padding_id
        self.w[:], self.seq_len[:] = 0, 0
        for i in range(len(self.l)):
            for j in range(len(self.l[0])):
                self.l[i][j] = None
        # populating new data
        for i_batch in range(self._batch_size):
            if self._read_sentences[i_batch] >= self._distances[i_batch]:
                self._read_tokens[i_batch] = -1
                continue
            cur_pos = self._batch_sen_idx[self._pointers[i_batch]]
            cur_pos += self._read_tokens[i_batch]
            for i_step in range(self._num_steps):
                if self._read_tokens[i_batch] == -1:
                    break
                self.x[i_batch, i_step] = self._data_x[cur_pos + i_step]
                self.y[i_batch, i_step] = self._data_y[cur_pos + i_step + 1]
                self.w[i_batch, i_step] = 1
                self.seq_len[i_batch] += 1
                self._read_tokens[i_batch] += 1
                self.l[i_batch][i_step] = self._find_label(cur_pos + i_step)
                if self._data[cur_pos + i_step + 1] == self._vocab.eos_id:
                    self._read_tokens[i_batch] = -1
        return self.x, self.y, self.w, self.l, self.seq_len

    def is_new_sen(self):
        return self._new_sentence_set

######################################################
# Module Functions
######################################################

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

def map_vocab_defs(vocab_filepath, def_prep_dir, out_filepath):
    vocab_t = Vocabulary.from_vocab_file(vocab_filepath)
    vocab_d = Vocabulary.from_vocab_file(
        os.path.join(def_prep_dir, 'vocab.txt'))
    definitions = {}
    max_len = 0
    lines = 0
    with open(os.path.join(def_prep_dir, 'train.jsonl')) as ifp:
        for line in ifp:
            lines += 1
            e = json.loads(line)
            word = e['meta']['word']
            defi = e['lines'][0].split()
            if word not in definitions:
                definitions[word] = []
            definitions[word].append(defi)
            if len(defi) > max_len:
                max_len = len(defi)
    data = np.zeros([lines + 1, max_len], np.int32)
    data[:] = vocab_d.eos_id
    index = np.zeros([vocab_t.vocab_size + 1], np.int32)
    for i in range(vocab_t.vocab_size):
        w = vocab_t.i2w(i)
        if w in definitions:
            defs = definitions[w]
            index[i+1] = index[i] + len(defs)
            for j in range(len(defs)):
                for k, t in enumerate(defs[j]):
                    data[index[i] + j, k] = vocab_d.w2i(defs[j][k])
        else:
            index[i+1] = index[i]
    with open(out_filepath, 'w') as ofp:
        cPickle.dump((index, data), ofp)
