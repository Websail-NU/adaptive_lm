"""Common functions

Todo:
    - Colorize log http://plumberjack.blogspot.com/2010/12/colorizing-logging-output-in-terminals.html

"""
import neobunch
import logging
import argparse
import collections
import json
import os
import sys

class LazyBunch(neobunch.Bunch):
    """ Just like Bunch,
        but return None if a requested attribute is not defined.
    """
    def __getattr__(self, k):
        try:
            return super(LazyBunch, self).__getattr__(k)
        except AttributeError:
            return None

    def toPretty(self):
        keys = self.keys()
        keys = sorted(keys)
        o = []
        for k in keys:
            o.append("{}:\t{}".format(k, self[k]))
        return '\n'.join(o)

    def toPrettyJSON(self, indent=2, sort_keys=True):
        return json.dumps(self.toDict(), indent=indent, sort_keys=sort_keys)

    @staticmethod
    def fromNeoBunch(nb):
        lb = LazyBunch(nb)
        return lb

    @staticmethod
    def fromDict(d):
        return LazyBunch.fromNeoBunch(
            super(LazyBunch, LazyBunch).fromDict(d))

    @staticmethod
    def fromNamespace(ns):
        return LazyBunch.fromDict(vars(ns))

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
    return LazyBunch(
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', dest='gpu',
                        action='store_true')
    parser.add_argument('--no-gpu', dest='gpu',
                        action='store_false')
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
    parser.add_argument('--training', dest='training',
                        action='store_true',
                        help='Set to train model.')
    parser.add_argument('--no-training', dest='training',
                        action='store_false',
                        help='Set to only run model.')
    parser.set_defaults(training=False)
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
    parser.add_argument('--no-varied_len', dest='varied_len',
                        action='store_false',
                        help='use static RNN graph')
    parser.set_defaults(varied_len=False)
    parser.add_argument('--sen_independent', dest='sen_independent',
                        action='store_true',
                        help=('Training RNN with padded batch, '
                              'and reset state every time a batch '
                              'comes from new sentences.'))
    parser.add_argument('--no-sen_independent', dest='sen_independent',
                        action='store_false',
                        help='Maintain RNN state')
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
    parser.add_argument('--experiment_dir', type=str, default='experiments/out',
                        help=('directory to store final and'
                              ' intermediate results and models.'))
    parser.add_argument('--log_file', type=str, default='experiment.log',
                        help='log file in experiment_dir')
    parser.add_argument('--load_config_filepath', type=str, default=None,
                        help=('Configuration absolute filepath template.'
                              'Settings will overrided by current arguments.'))
    parser.add_argument('--save_config_file', type=str, default=None,
                        help='Configuration file to save in experiment_dir.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='display debug information')
    parser.add_argument('--no-debug', dest='debug', action='store_false',
                        help='no display debug information')
    parser.set_defaults(debug=False)
    parser.add_argument('--progress_steps', type=int,
                        default=1000,
                        help='frequency for progress report in training')
    return parser

def update_opt(opt, parser):
    args = parser.parse_args()
    new_opt = LazyBunch.fromNamespace(args)
    if args.load_config_filepath is not None:
        with open(args.load_config_filepath) as ifp:
            opt.update(LazyBunch.fromDict(json.load(ifp)))
        for arg in sys.argv:
            if arg.startswith('--') and len(arg) > 2:
                arg = arg[2:]
                if arg.startswith('no-'):
                    opt[arg[3:]] = new_opt[arg[3:]]
                else:
                    opt[arg] = new_opt[arg]
    else:
        opt.update(new_opt)
    return opt

def save_config_file(opt):
    p = os.path.join(opt.experiment_dir, opt.save_config_file)
    with open(p, 'w') as ofp:
        ofp.write(opt.toPrettyJSON())

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
