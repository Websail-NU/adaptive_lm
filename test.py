""" Testing script

This module loads and tests a recurrent neural network language model.

"""

import argparse
import logging
import time
import os
import json

import numpy as np
import tensorflow as tf

import lm
import common_utils
import data_utils
from exp_utils import *

def main(opt):
    vocab_path = os.path.join(opt.data_dir, opt.vocab_file)
    test_path = os.path.join(opt.data_dir, opt.test_file)
    logger.info('Loading data set...')
    logger.debug('- Loading vocab from {}'.format(vocab_path))
    vocab = data_utils.Vocabulary.from_vocab_file(vocab_path)
    logger.debug('-- vocab size: {}'.format(vocab.vocab_size))
    logger.debug('- Loading test data from {}'.format(test_path))
    if opt.sen_independent:
        test_iter = data_utils.SentenceIterator(vocab, test_path)
    else:
        test_iter = data_utils.DataIterator(vocab, test_path)
    opt.vocab_size = vocab.vocab_size
    logger.debug('Staring session...')
    # with tf.device("/cpu:0"), tf.Session() as sess:
    with tf.Session() as sess:
        logger.info('Creating model...')
        init_scale = opt.init_scale
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating a model...')
        if opt.shared_emb:
            with tf.variable_scope('shared_emb'):
                shared_emb_vars = lm.sharded_variable(
                    'emb', [opt.vocab_size, opt.emb_size], opt.num_shards)
                opt.input_emb_vars = shared_emb_vars
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            model = lm.LM(opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        state = common_utils.get_initial_training_state()
        state.learning_rate = opt.learning_rate
        state, success = resume_if_possible(opt, sess, saver, state,
                                            prefix="best")
        if not success:
            logger.error('Failed to load the model. Testing aborted.')
            return
        logger.info('Testing...')
        token_loss = None
        if opt.out_token_loss_file is not None:
            token_loss = []
        ppl, steps = run_epoch(sess, model, test_iter, opt,
                               token_loss=token_loss)
        logger.info('PPL = {}'.format(ppl))
        if token_loss is not None:
            logger.info('Writing token loss...')
            token_loss_path = os.path.join(opt.output_dir, opt.out_token_loss_file)
            with open(token_loss_path, 'w') as ofp:
                for p in token_loss:
                    ofp.write("{}\t{}\n".format(vocab.i2w(p[0]), p[1]))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--out_token_loss_file', type=str,
                        default=None,
                        help='output token loss to the file')
    parser.add_argument('--shared_emb', dest='shared_emb',
                        action='store_true', help='use emb from shared_emb scope')
    parser.set_defaults(shared_emb=False)
    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)
    logger = common_utils.get_logger(opt.log_file_path)
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.__repr__()))
    if opt.out_token_loss_file is not None:
        if opt.batch_size > 1:
            logger.warn(("Batch size is larger than 1."
                         "Output token loss will not align."))
        if opt.num_steps > 1:
            logger.warn(("Num steps is larger than 1."
                         "Output token may loss toward the end."))
    main(opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
