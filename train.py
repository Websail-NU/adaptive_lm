""" Training script

This module creates and trains recurrent neural network language model.

Example:

Todo:
    * Implement learning rate decay
    * Implement ealry stop

"""

import argparse
import logging
import time
import os
import json

import tensorflow as tf
import numpy as np

import lm
import common_utils
import data_utils

def run_epoch(sess, m, data_iter, opt, train_op=tf.no_op()):
    """ train the model on the given data. """
    start_time = time.time()
    costs = 0.0
    num_words = 0
    state = []
    for c, h in m.initial_state:
        state.append((c.eval(), h.eval()))
    for step, (x, y, w, l) in enumerate(data_iter.iterate_epoch(
        m.opt.batch_size, m.opt.num_steps)):
        fetches = [m.loss, train_op]
        for c, h in m.final_state:
            fetches.append(c)
            fetches.append(h)
        feed_dict = {m.x: x, m.y: y, m.w: w}
        for i, (c, h) in enumerate(m.initial_state):
            feed_dict[c], feed_dict[h] = state[i]
        res = sess.run(fetches, feed_dict)
        cost = res[0]
        state_flat = res[2:]
        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        costs += cost
        num_words += np.sum(w)
        if train_op.name != u'NoOp' and (step + 1) % opt.progress_steps == 0:
            logger.info("-- @{} perplexity: {} wps: {}".format(
                    step + 1, np.exp(costs / (step + 1)),
                    num_words / (time.time() - start_time)))
    return np.exp(costs / (step+1)), step

def main(opt):
    vocab_path = os.path.join(opt.data_dir, opt.vocab_file)
    train_path = os.path.join(opt.data_dir, opt.train_file)
    valid_path = os.path.join(opt.data_dir, opt.valid_file)
    logger.info('Loading data set...')
    logger.debug('- Loading vocab from {}'.format(vocab_path))
    vocab = data_utils.Vocabulary.from_vocab_file(vocab_path)
    logger.debug('-- vocab size: {}'.format(vocab.vocab_size))
    logger.debug('- Loading train data from {}'.format(train_path))
    train_iter = data_utils.DataIterator(vocab, train_path)
    logger.debug('- Loading valid data from {}'.format(valid_path))
    valid_iter = data_utils.DataIterator(vocab, valid_path)
    opt.vocab_size = vocab.vocab_size
    logger.debug('Staring session...')
    with tf.Session() as sess:
        logger.info('Creating model...')
        init_scale = opt.init_scale
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating training model...')
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = lm.LM(opt)
            train_op, lr_var = lm.train_op(model, model.opt)
            logger.debug('- Creating validating model (reuse params)...')
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            vmodel = lm.LM(opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())
        state = common_utils.get_initial_training_state()
        state.learning_rate = opt.learning_rate
        for epoch in range(opt.max_epochs):
            state.epoch = epoch
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lr_var, state.learning_rate))
            logger.info("- Learning rate = {}".format(state.learning_rate))
            logger.info('- Training:')
            train_ppl, steps = run_epoch(sess, model, train_iter, opt, train_op)
            logger.info('- Validating:')
            valid_ppl, vsteps = run_epoch(sess, vmodel, valid_iter, opt)
            logger.info('- Train ppl = {}, Valid ppl = {}'.format(
                train_ppl, valid_ppl))
            logger.info('========= End of epoch {} report ========='.format(
                epoch + 1))
            if valid_ppl < state.best_val_ppl:
                logger.info('- Best PPL: {} -> {}'.format(
                    state.best_val_ppl, valid_ppl))
                logger.info('- Epoch: {} -> {}'.format(
                    state.best_epoch + 1, epoch + 1))
                state.best_val_ppl = valid_ppl
                state.best_epoch = epoch
                checkpoint_path = os.path.join(opt.output_dir,
                                               "best_model.ckpt")
                logger.info('- Saving model to {}'.format(checkpoint_path))
                save_path = saver.save(sess, checkpoint_path)
                with open(os.path.join(opt.output_dir,
                                       "best_state.json"), 'w') as ofp:
                    json.dump(vars(state), ofp)
            else:
                logger.info('- No improvement!')

if __name__ == "__main__":
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
