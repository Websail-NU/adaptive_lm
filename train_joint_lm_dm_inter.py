import argparse
import logging
import time
import os
import json
import random
import cPickle
# random.seed(1234)

import numpy as np
# np.random.seed(1234)
import tensorflow as tf
# tf.set_random_seed(1234)

import lm
import common_utils
import data_utils
from exp_utils import *

def main(lm_opt, dm_opt):
    prefix = ['latest_lm', 'latest_dm']
    dataset = ['train', 'valid']
    shortlist_vocab_path = lm_opt.shortlist_path
    dm_emb_path = os.path.join(dm_opt.data_dir, 'emb.cpickle')
    logger.debug('- Loading shortlist vocab from {}'.format(
        shortlist_vocab_path))
    short_vocab = data_utils.Vocabulary.from_vocab_file(shortlist_vocab_path)
    logger.debug('-- Shortlist vocab size: {}'.format(short_vocab.vocab_size))
    lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset,
                                      y_vocab=short_vocab)
    dm_data, dm_vocab = load_datasets(dm_opt, dataset=dataset,
                                      iterator_type=data_utils.SenLabelIterator,
                                      l_vocab=short_vocab)
    lm_opt.vocab_size = lm_vocab.vocab_size
    dm_opt.vocab_size = dm_vocab.vocab_size
    lm_vocab_mask = data_utils.Vocabulary.create_vocab_mask(
        lm_vocab, short_vocab)
    lm_opt.logit_mask = lm_vocab_mask

    logger.info('Loading data completed')

    init_scale = lm_opt.init_scale
    sess_config =tf.ConfigProto(log_device_placement=False)
                                # device_count = {'GPU': 0})
    logger.info('Starting TF Session...')

    with tf.Session(config=sess_config) as sess:
        logger.debug(
                '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating shared embedding variables...')
        with tf.variable_scope('shared_emb'):
            shared_emb_vars = lm.sharded_variable(
                'emb', [short_vocab.vocab_size, lm_opt.emb_size],
                lm_opt.num_shards)
        logger.debug('- Loading embedding for DM...')
        with open(dm_emb_path) as ifp:
            emb_array = cPickle.load(ifp)
            dm_emb_init = tf.constant(emb_array, dtype=tf.float32)
        lm_opt.softmax_w_vars = shared_emb_vars
        dm_opt.af_ex_emb_vars = shared_emb_vars
        dm_opt.input_emb_init = dm_emb_init
        dm_opt.input_emb_trainable = False
        logger.debug('- Creating training LM...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            lm_train = lm.LM(lm_opt)
            lm_train_op, lm_lr_var = lm.train_op(lm_train, lm_opt)
        logger.debug('- Creating validating LM (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            lm_valid = lm.LM(lm_opt, is_training=False)
        logger.debug('- Creating training DM...')
        with tf.variable_scope('DM', reuse=None, initializer=initializer):
            dm_train = lm.LMwAF(dm_opt)
            dm_train_op, dm_lr_var = lm.train_op(dm_train, dm_opt)
        logger.debug('- Creating validating DM (reuse params)...')
        with tf.variable_scope('DM', reuse=True, initializer=initializer):
            dm_valid = lm.LMwAF(dm_opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        states = {}
        for p in prefix:
            states[p] = common_utils.get_initial_training_state()
        states, _ = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, prefix)
        lm_state = states[prefix[0]]
        dm_state = states[prefix[1]]
        lm_state.learning_rate = lm_opt.learning_rate
        dm_state.learning_rate = dm_opt.learning_rate

        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())

        for epoch in range(lm_state.epoch, lm_opt.max_epochs):
            epoch_time = time.time()
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lm_lr_var, lm_state.learning_rate))
            sess.run(tf.assign(dm_lr_var, dm_state.learning_rate))
            logger.info("- Traning DM with learning rate {}...".format(
                dm_state.learning_rate))
            dm_train_ppl, _ = run_epoch(sess, dm_train, dm_data['train'],
                                        dm_opt, train_op=dm_train_op)
            logger.info('- Validating DM...')
            dm_valid_ppl, _ = run_epoch(sess, dm_valid,
                                        dm_data['valid'], dm_opt)
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
            logger.info('----------------------------------')
            logger.info('DM post epoch routine...')
            run_post_epoch(dm_train_ppl, dm_valid_ppl, dm_state, dm_opt,
                           best_prefix="best_dm",
                           latest_prefix="latest_dm")
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            if done_training:
                break
        logger.info('Done training at epoch {}'.format(lm_state.epoch + 1))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--af_mode', type=str, default='gated_state',
                        help='additional feature module type')
    parser.add_argument('--def_dir', type=str,
                        default='data/common_defs_v1.2/wordnet/preprocess',
                        help='data directory for dictionary corpus')
    parser.add_argument('--shortlist_path', type=str,
                        default=('data/common_defs_v1.2/wordnet/shortlist/'
                                 'shortlist_all_ptb.txt'),
                        help='vocab file for shared emb')
    args = parser.parse_args()
    lm_opt = common_utils.Bunch.default_model_options()
    lm_opt.update_from_ns(args)
    dm_opt = common_utils.Bunch.default_model_options()
    dm_opt.update_from_ns(args)
    dm_opt.af_function = 'ex_emb'
    dm_opt.data_dir = dm_opt.def_dir
    dm_opt.sen_independent = True
    logger = common_utils.get_logger(lm_opt.log_file_path)
    if lm_opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(dm_opt.__repr__()))
    main(lm_opt, dm_opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
