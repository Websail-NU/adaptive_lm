import argparse
import logging
import time
import os
import json
import random
random.seed(1234)

import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)

import lm
import common_utils
import data_utils
from exp_utils import *

def main(opt_lm, opt_dm):
    vocab_lm_path = os.path.join(opt_lm.data_dir, opt_lm.vocab_file)
    train_lm_path = os.path.join(opt_lm.data_dir, opt_lm.train_file)
    valid_lm_path = os.path.join(opt_lm.data_dir, opt_lm.valid_file)
    vocab_dm_path = os.path.join(opt_dm.data_dir, opt_dm.vocab_file)
    train_dm_path = os.path.join(opt_dm.data_dir, opt_dm.train_file)
    shortlist_path = os.path.join(opt_dm.data_dir, "train_shortlist.txt")
    logger.info('Loading data set...')
    logger.debug('- Loading vocab LM from {}'.format(vocab_lm_path))
    vocab_lm = data_utils.Vocabulary.from_vocab_file(vocab_lm_path)
    logger.debug('- LM vocab size: {}'.format(vocab_lm.vocab_size))
    logger.debug('- Loading vocab DM from {}'.format(vocab_dm_path))
    vocab_dm = data_utils.Vocabulary.from_vocab_file(vocab_dm_path)
    logger.debug('- DM vocab size: {}'.format(vocab_dm.vocab_size))
    logger.debug('- Loading train LM data from {}'.format(train_lm_path))
    train_lm_iter = data_utils.DataIterator(vocab_lm, train_lm_path)
    logger.debug('- Loading valid LM data from {}'.format(valid_lm_path))
    valid_lm_iter = data_utils.DataIterator(vocab_lm, valid_lm_path)
    logger.debug('- Loading train DM data from {}'.format(train_dm_path))
    train_dm_iter = data_utils.DefIterator(vocab_dm, train_dm_path)
    opt_dm.num_steps = train_dm_iter._max_seq_len
    logger.debug('- Loading shortlist ID from {}'.format(shortlist_path))
    shortlist_lm = data_utils.Vocabulary.list_ids_from_file(
        shortlist_path, vocab_lm)
    shortlist_dm = data_utils.Vocabulary.list_ids_from_file(
        shortlist_path, vocab_dm)
    lm2dm, dm2lm = data_utils.Vocabulary.vocab_index_map(vocab_lm, vocab_dm)
    opt_lm.vocab_size = vocab_lm.vocab_size
    opt_dm.vocab_size = vocab_dm.vocab_size
    init_scale = opt_lm.init_scale
    logger.info('Loading data completed')
    logger.debug('- LM voacb size: {}'.format(vocab_lm.vocab_size))
    logger.debug('- DM voacb size: {}'.format(vocab_dm.vocab_size))
    logger.debug('- Shared voacb size: {}'.format(len(shortlist_lm)))

    sess_config =tf.ConfigProto(log_device_placement=False)
    logger.info('Start TF Session')
    # tf.device('/gpu:0'),
    with tf.Session(config=sess_config) as sess:
        logger.debug(
                '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating training LM...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            train_lm = lm.LM(opt_lm)
            train_lm_op, lr_lm_var = lm.train_op(train_lm, train_lm.opt)
        logger.debug('- Creating validating LM (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            valid_lm = lm.LM(opt_lm, is_training=False)
        logger.debug('- Creating training DM ...')
        with tf.variable_scope('DM', reuse=None, initializer=initializer):
            train_dm = lm.LMwAF(opt_dm)
            train_dm_op, lr_dm_var = lm.train_op(train_dm, train_dm.opt)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        state = common_utils.get_initial_training_state()
        state.learning_rate = opt_lm.learning_rate
        state, _ = resume_if_possible(opt_lm, sess, saver, state)
        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())

        for epoch in range(state.epoch, opt_lm.max_epochs):
            epoch_time = time.time()
            state.epoch = epoch
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lr_lm_var, state.learning_rate))
            sess.run(tf.assign(lr_dm_var, state.learning_rate))
            logger.info("- Learning rate = {}".format(state.learning_rate))
            logger.debug("- Leanring rate (variable) = {}".format(
                sess.run(lr_lm_var)))
            train_dm_ppl = 0
            if opt_dm.train_dm:
                logger.info("Traning DM...")
                transfer_emb(sess, "LM", "DM", shortlist_lm, lm2dm)
                train_dm_ppl, dsteps = run_epoch(sess, train_dm, train_dm_iter,
                                                 opt_dm, train_dm_op)
                transfer_emb(sess, "DM", "LM", shortlist_dm, dm2lm)
            logger.info("Traning LM...")
            train_lm_ppl, lsteps = run_epoch(sess, train_lm, train_lm_iter,
                                             opt_lm, train_lm_op)
            logger.info("Validating LM...")
            valid_lm_ppl, vsteps = run_epoch(sess, valid_lm, valid_lm_iter, opt_lm)
            logger.info('- DM ppl = {}, Train ppl = {}, Valid ppl = {}'.format(
                        train_dm_ppl, train_lm_ppl, valid_lm_ppl))
            # print('{}\t{}\t{}\t{}'.format(
            #     epoch+1, train_dm_ppl, train_lm_ppl, valid_lm_ppl))
            logger.info('Post epoch routine...')
            state.val_ppl = valid_lm_ppl
            if valid_lm_ppl < state.best_val_ppl:
                logger.info('- Best PPL: {} -> {}'.format(
                    state.best_val_ppl, valid_lm_ppl))
                logger.info('- Epoch: {} -> {}'.format(
                    state.best_epoch + 1, epoch + 1))
                state.best_val_ppl = valid_lm_ppl
                state.best_epoch = epoch
                ckpt_path = os.path.join(opt_lm.output_dir, "best_model.ckpt")
                state_path = os.path.join(opt_lm.output_dir, "best_state.json")
                logger.info('- Saving best model to {}'.format(ckpt_path))
                saver.save(sess, ckpt_path)
                with open(state_path, 'w') as ofp:
                    json.dump(vars(state), ofp)
            else:
                logger.info('- No improvement!')
            done_training = update_lr(opt_lm, state)
            ckpt_path = os.path.join(opt_lm.output_dir, "latest_model.ckpt")
            state_path = os.path.join(opt_lm.output_dir, "latest_state.json")
            logger.info('End of epoch {}: '.format(
                epoch + 1))
            logger.info('- Saving model to {}'.format(ckpt_path))
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            saver.save(sess, ckpt_path)
            with open(state_path, 'w') as ofp:
                json.dump(vars(state), ofp)
            if done_training:
                break
            logger.debug('Updated state:\n{}'.format(state.__repr__()))
        logger.info('Done training at epoch {}'.format(state.epoch + 1))


if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--train_dm', dest='train_dm',
                        action='store_true', help='train DM')
    parser.set_defaults(train_dm=False)
    parser.add_argument('--af_mode', type=str, default='gated_state',
                        help='additional feature module type')
    args = parser.parse_args()
    opt_lm = common_utils.Bunch.default_model_options()
    opt_lm.update_from_ns(args)
    opt_dm = common_utils.Bunch.default_model_options()
    opt_dm.update_from_ns(args)
    # With GPU, this will slow us down.
    # A proper weights to the loss function is enough to get correct gradients
    # opt_dm.varied_len = True
    opt_dm.reset_state = True
    opt_dm.data_dir = "data/ptb_defs/preprocess"
    logger = common_utils.get_logger(opt_lm.log_file_path)
    if opt_lm.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt_dm.__repr__()))
    main(opt_lm, opt_dm)
    logger.info('Total time: {}s'.format(time.time() - global_time))
