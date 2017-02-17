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

def main(opt):
    vocab_path = os.path.join(opt.data_dir, opt.vocab_file)
    train_path = os.path.join(opt.data_dir, opt.train_file)
    valid_path = os.path.join(opt.data_dir, opt.valid_file)
    vocab_emb_path = opt.shared_emb_vocab
    logger.info('Loading data set...')
    logger.debug('- Loading vocab from {}'.format(vocab_path))
    vocab = data_utils.Vocabulary.from_vocab_file(vocab_path)
    logger.debug('-- vocab size: {}'.format(vocab.vocab_size))
    logger.debug('- Loading vocab shared emb from {}'.format(vocab_emb_path))
    vocab_emb= data_utils.Vocabulary.from_vocab_file(vocab_emb_path)
    logger.debug('-- Shared emb vocab size: {}'.format(vocab_emb.vocab_size))
    logger.debug('-- vocab size: {}'.format(vocab.vocab_size))
    logger.debug('- Loading train data from {}'.format(train_path))
    logger.debug('- Loading valid data from {}'.format(valid_path))
    if opt.sen_independent:
        train_iter = data_utils.SentenceIterator(vocab, train_path)
        valid_iter = data_utils.SentenceIterator(vocab, valid_path)
    else:
        train_iter = data_utils.DataIterator(vocab, train_path)
        valid_iter = data_utils.DataIterator(vocab, valid_path)
    opt.vocab_size = vocab.vocab_size
    if opt.shared_emb_lm_logit:
        logger.debug('-- Vocab mask detected, reloading LM data...')
        lm_vocab_mask = data_utils.Vocabulary.create_vocab_mask(vocab, vocab_emb)
        if opt.sen_independent:
            train_iter = data_utils.SentenceIterator(vocab_emb, train_path)
            valid_iter = data_utils.SentenceIterator(vocab_emb, valid_path)
        else:
            train_iter = data_utils.DataIterator(vocab_emb, train_path)
            valid_iter = data_utils.DataIterator(vocab_emb, valid_path)
        opt.vocab_size = vocab_emb.vocab_size
        opt.logit_mask = lm_vocab_mask
    logger.info('Loading data completed')

    logger.debug('Staring session...')
    with tf.Session() as sess:
        logger.info('Creating model...')
        init_scale = opt.init_scale
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating shared embedding variables...')
        if opt.shared_emb_lm_logit:
            with tf.variable_scope('shared_emb'):
                shared_emb_vars = lm.sharded_variable(
                    'emb', [vocab_emb.vocab_size, opt.emb_size],
                    opt.num_shards)
            opt.input_emb_vars = shared_emb_vars
            opt.softmax_w_vars = shared_emb_vars
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
        state = common_utils.get_initial_training_state()
        state.learning_rate = opt.learning_rate
        state, _ = resume_if_possible(opt, sess, saver, state)
        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())
        for epoch in range(state.epoch, opt.max_epochs):
            epoch_time = time.time()
            state.epoch = epoch
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lr_var, state.learning_rate))
            logger.info("- Learning rate = {}".format(state.learning_rate))
            logger.debug("- Leanring rate (variable) = {}".format(
                sess.run(lr_var)))
            logger.info('- Training:')
            train_ppl, steps = run_epoch(sess, model, train_iter, opt,
                                         train_op=train_op)
            logger.info('- Validating:')
            valid_ppl, vsteps = run_epoch(sess, vmodel, valid_iter, opt)
            logger.info('- Train ppl = {}, Valid ppl = {}'.format(
                train_ppl, valid_ppl))
            state.val_ppl = valid_ppl
            if valid_ppl < state.best_val_ppl:
                logger.info('- Best PPL: {} -> {}'.format(
                    state.best_val_ppl, valid_ppl))
                logger.info('- Epoch: {} -> {}'.format(
                    state.best_epoch + 1, epoch + 1))
                state.best_val_ppl = valid_ppl
                state.best_epoch = epoch
                ckpt_path = os.path.join(opt.output_dir, "best_model.ckpt")
                state_path = os.path.join(opt.output_dir, "best_state.json")
                logger.info('- Saving best model to {}'.format(ckpt_path))
                saver.save(sess, ckpt_path)
                with open(state_path, 'w') as ofp:
                    json.dump(vars(state), ofp)
            else:
                logger.info('- No improvement!')
            done_training = update_lr(opt, state)
            ckpt_path = os.path.join(opt.output_dir, "latest_model.ckpt")
            state_path = os.path.join(opt.output_dir, "latest_state.json")
            logger.info('--------- End of epoch {} ---------'.format(
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
    parser.add_argument('--shared_emb_vocab', type=str,
                        default='data/ptb/preprocess/vocab.txt',
                        help='vocab file for shared emb')
    parser.add_argument('--shared_emb_lm_logit', dest='shared_emb_lm_logit',
                        action='store_true',
                        help=('use shared emb for lm logit weight'))
    parser.set_defaults(shared_emb_lm_logit=False)
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
