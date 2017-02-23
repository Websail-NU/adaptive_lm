"""Common functions

Todo:
    - Colorize log http://plumberjack.blogspot.com/2010/12/colorizing-logging-output-in-terminals.html

"""

import logging
import argparse
import collections

class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def update_from_dict(self, d):
        # print(unicode2string(d))
        self.__dict__.update(d)

    def update_from_ns(self, ns):
        """Works with argparse"""
        self.update_from_dict(vars(ns))

    def is_set(self, attr):
        return attr in self.__dict__

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        keys = self.__dict__.keys()
        keys = sorted(keys)
        o = []
        for k in keys:
            o.append("- {}:\t{}".format(k, self.__dict__[k]))
        return '\n'.join(o)

    @staticmethod
    def default_model_options():
        opt = Bunch()
        opt.__dict__.update(
            batch_size=32,
            num_steps=10,
            num_shards=1,
            num_layers=1,
            varied_len=False,
            learning_rate=0.5,
            max_grad_norm=10.0,
            emb_keep_prob=0.9,
            keep_prob=0.75,
            vocab_size=10000,
            emb_size=100,
            state_size=100,
            num_softmax_sampled=0,
            run_profiler=False,
            init_scale=0.1,
            input_emb_trainable=True
        )
        return opt

def unicode2string(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(unicode2string, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(unicode2string, data))
    else:
        return data

def get_logger(log_file_path=None, name="exp"):
    root_logger = logging.getLogger(name)
    if log_file_path is not None:
        log_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt='%Y/%m/%d %H:%M:%S')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    time_format = "\x1b[36m%(asctime)s\x1b[0m"
    level_format = "\x1b[94m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = logging.Formatter(
        "{} {} %(message)s".format(time_format, level_format),
        datefmt='%Y/%m/%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger

def SUN_BRO():
    return '\[T]/ PRAISE THE SUN!\n |_|\n | |'

def get_initial_training_state():
    return Bunch(
        epoch = 0,
        val_ppl = float('inf'),
        best_val_ppl = float('inf'),
        learning_rate = 0.01,
        best_epoch = -1,
        last_imp_val_ppl = float('inf'),
        last_imp_epoch = -1,
        imp_wait = 0
    )

def get_tf_sess_config(opt):
    import tensorflow as tf
    sess_config = tf.ConfigProto(log_device_placement=False,
                                 device_count = {'GPU': 0})
    if opt.gpu:
        sess_config = tf.ConfigProto(log_device_placement=False)
    return sess_config

def get_common_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu',
                        action='store_true')
    parser.set_defaults(gpu=False)
    # Data and vocabulary file
    parser.add_argument('--data_dir', type=str,
                        default='data/ptb/preprocess',
                        help='data directory')
    parser.add_argument('--train_file', type=str,
                        default='train.jsonl',
                        help='train data file')
    parser.add_argument('--valid_file', type=str,
                        default='valid.jsonl',
                        help='valid data file')
    parser.add_argument('--test_file', type=str,
                        default='test.jsonl',
                        help='test data file')
    parser.add_argument('--vocab_file', type=str,
                        default='vocab.txt',
                        help='vocab file')
    # Parameters to configure the neural network.
    parser.add_argument('--state_size', type=int, default=100,
                        help='size of RNN hidden state vector')
    parser.add_argument('--emb_size', type=int, default=100,
                        help='size of character embeddings')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='number of unrolling steps.')
    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='initialized value for model params')
    # parser.add_argument('--model', type=str, default='lstm',
    #                     help='which model to use (rnn, lstm or gru).')

    # Parameters to control the training.
    parser.add_argument('--optim', type=str, default="sgd",
                        help='Optimization algorithm: sgd or adam')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='number of maxinum epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--emb_keep_prob', type=float, default=0.9,
                        help='(1 - dropout probability) of the embedding')
    parser.add_argument('--keep_prob', type=float, default=0.75,
                        help=('(1 - dropout probability)'
                              'of other part of the model'))
    parser.add_argument('--varied_len', dest='varied_len', action='store_true',
                        help=('create dynamic RNN graph which will not compute '
                              'the RNN steps past the sequence length. '
                              'You should avoid setting this to true '
                              'if input is always in full sequence.'))
    parser.set_defaults(varied_len=False)
    parser.add_argument('--reset_state', dest='reset_state',
                        action='store_true',
                        help=('Reset RNN state for each minibatch, '
                              '(always reset state every epoch).'))
    parser.set_defaults(reset_state=False)
    parser.add_argument('--sen_independent', dest='sen_independent',
                        action='store_true',
                        help=('Training RNN with padded batch, '
                              'and reset state every time a batch '
                              'comes from new sentences.'))
    parser.set_defaults(sen_independent=False)
    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=10.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='initial learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--lr_decay_every', type=int, default=-1,
                        help='number of epochs before learning rate is decayed')
    parser.add_argument('--lr_decay_imp', type=float, default=0.96,
                        help=('improvement ratio between val losses before'
                              'decaying learning rate'))
    parser.add_argument('--lr_decay_wait', type=int, default=2,
                        help=('number of non-improving epochs to wait'
                              'before decaying learning rate'))
    parser.add_argument('--lr_decay_factor', type=float, default=0.8,
                        help=('factor by which learning rate'
                              'is decayed (lr = lr * factor)'))
    # Parameters for outputs and reporting.
    parser.add_argument('--output_dir', type=str, default='data/ptb/output',
                        help=('directory to store final and'
                              ' intermediate results and models.'))
    parser.add_argument('--log_file_path', type=str, default=None,
                        help='log file')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='display debug information')
    parser.set_defaults(debug=False)
    parser.add_argument('--progress_steps', type=int,
                        default=1000,
                        help='frequency for progress report in training')
    return parser
