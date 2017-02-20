""" Training script

This module creates and trains recurrent neural network language model.

Example:

Todo:
    - Support other optimization method
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

def main(lm_opt):
    model_prefix = ['latest_lm']
    dataset = ['train', 'valid']
    lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset)
    lm_opt.vocab_size = lm_vocab.vocab_size
    logger.info('Loading data completed')
    init_scale = lm_opt.init_scale
    sess_config = common_utils.get_tf_sess_config(lm_opt)
    logger.info('Starting TF Session...')

    with tf.Session(config=sess_config) as sess:
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating training LM...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            lm_train = lm.LM(lm_opt)
            lm_train_op, lm_lr_var = lm.train_op(lm_train, lm_opt)
        logger.debug('- Creating validating LM (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            lm_valid = lm.LM(lm_opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        states = {}
        for p in model_prefix:
            states[p] = common_utils.get_initial_training_state()
        states, _ = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, model_prefix)
        lm_state = states[model_prefix[0]]
        lm_state.learning_rate = lm_opt.learning_rate

        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())

        for epoch in range(lm_state.epoch, lm_opt.max_epochs):
            epoch_time = time.time()
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lm_lr_var, lm_state.learning_rate))
            logger.info("- Traning LM with learning rate {}...".format(
                lm_state.learning_rate))
            lm_train_ppl, _ = run_epoch(sess, lm_train, lm_data['train'],
                                        lm_opt, train_op=lm_train_op)
            logger.info('- Validating LM...')
            lm_valid_ppl, _ = run_epoch(sess, lm_valid,
                                        lm_data['valid'], lm_opt)
            logger.info('----------------------------------')
            logger.info('LM post epoch routine...')
            done_training = run_post_epoch(
                lm_train_ppl, lm_valid_ppl, lm_state, lm_opt,
                sess=sess, saver=saver,
                best_prefix="best_lm", latest_prefix="latest_lm")
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            if done_training:
                break
        logger.info('Done training at epoch {}'.format(lm_state.epoch + 1))

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
