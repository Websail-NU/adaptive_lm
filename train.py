""" Training script

This module creates and trains recurrent neural network language model.

Example:

Todo:
    - Support TensorBoard

"""

import argparse
import logging
import time
import os
import json
import random
# random.seed(1234)

import numpy as np
# np.random.seed(1234)
import tensorflow as tf
# tf.set_random_seed(1234)

import lm
import common_utils
import data_utils
from exp_utils import *

def main(opt):
    prefix = ['latest_lm', 'latest_dm']
    dataset = ['train', 'valid']
    data, vocab = load_datasets(opt, dataset=dataset)
    logger.info('Loading data completed')
    opt.vocab_size = vocab.vocab_size
    init_scale = opt.init_scale
    logger.debug('Staring session...')
    sess_config = common_utils.get_tf_sess_config(lm_opt)
    with tf.Session(config=sess_config) as sess:
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating training model...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            model = lm.LM(opt)
            train_op, lr_var = lm.train_op(model, model.opt)
        logger.debug('- Creating validating model (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            vmodel = lm.LM(opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        states = {}
        for p in prefix:
            states[p] = common_utils.get_initial_training_state()
        states, _ = resume_many_states(opt.output_dir, sess,
                                       saver, states, prefix)
        state = states[prefix[0]]
        state.learning_rate = opt.learning_rate
        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())
        for epoch in range(state.epoch, opt.max_epochs):
            epoch_time = time.time()
            state.epoch = epoch
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lr_var, state.learning_rate))
            logger.info("- Traning LM with learning rate {}...".format(
                state.learning_rate))
            train_ppl, _ = run_epoch(sess, model, data['train'], opt,
                                         train_op=train_op)
            logger.info('- Validating LM...')
            valid_ppl, _ = run_epoch(sess, vmodel, data['valid'], opt)
            logger.info('----------------------------------')
            logger.info('LM post epoch routine...')
            done_training = run_post_epoch(
                train_ppl, valid_ppl, state, opt,
                sess=sess, saver=saver,
                best_prefix="best_lm", latest_prefix="latest_lm")
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            if done_training:
                break
        logger.info('Done training at epoch {}'.format(state.epoch + 1))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)
    logger = common_utils.get_logger(opt.log_file_path)
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.__repr__()))
    main(opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
