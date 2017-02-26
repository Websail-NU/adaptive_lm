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

def main(lm_opt):
    model_prefix = ['best_lm']
    dataset = ['test']
    lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset)
    lm_opt.vocab_size = lm_vocab.vocab_size
    logger.info('Loading data completed')
    init_scale = lm_opt.init_scale
    sess_config = common_utils.get_tf_sess_config(lm_opt)
    logger.info('Starting TF Session...')
    with tf.Session(config=sess_config) as sess:
        logger.info('Creating model...')
        init_scale = lm_opt.init_scale
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating training LM...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            model = lm.LM(lm_opt, is_training=False)
        logger.debug('Trainable variables:')
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        states = {}
        for p in model_prefix:
            states[p] = common_utils.get_initial_training_state()
        _, success = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, model_prefix)
        if not success:
            logger.error('Failed to load the model. Testing aborted.')
            return
        logger.info('Testing...')
        token_loss = None
        if opt.out_token_loss_file is not None:
            token_loss = []
        ppl, _ = run_epoch(sess, model, lm_data['test'], lm_opt,
                           token_loss=token_loss)
        logger.info('PPL = {}'.format(ppl))
        if token_loss is not None:
            logger.info('Writing token loss...')
            token_loss_path = os.path.join(opt.output_dir,
                                           opt.out_token_loss_file)
            with open(token_loss_path, 'w') as ofp:
                for p in token_loss:
                    ofp.write("{}\t{}\n".format(lm_vocab.i2w(p[0]), p[1]))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--out_token_loss_file', type=str,
                        default=None,
                        help='output token loss to the file')
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
