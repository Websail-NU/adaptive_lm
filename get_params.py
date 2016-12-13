""" Utility script

This script load a model checkpoint and load parameters into
numpy array

Example:

Todo:
    - Support other architecture than a standard LM
    - Refactor CLI options
"""
import os
import cPickle
import numpy as np
import tensorflow as tf

import lm
import common_utils
import data_utils
from exp_utils import *

def main(opt):
    with tf.Session() as sess:
        logger.info('Creating model...')
        init_scale = opt.init_scale
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            model = lm.LM(opt)
            train_op, lr_var = lm.train_op(model, model.opt)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        saver = tf.train.Saver()
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        state = common_utils.get_initial_training_state()
        state.learning_rate = opt.learning_rate
        state, _ = resume_if_possible(opt, sess, saver, state, prefix="best")
        trainable_vars = tf.trainable_variables()
        trainable_vals = sess.run(trainable_vars)
        params = {}
        for i, v in enumerate(trainable_vars):
            params[v.name] = trainable_vals[i]
        with open(os.path.join(opt.output_dir, opt.output_pickle_file), 'w') as ofp:
            cPickle.dump(params, ofp)
        return params

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--output_pickle_file', type=str,
                        default='params.pickle',
                        help=('Output pickle file for a dictionary of parameters'))
    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)
    logger = common_utils.get_logger(opt.log_file_path)
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.__repr__()))
    params = main(opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
