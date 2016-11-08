""" Training script

This module creates and trains recurrent neural network language model.

Example:

Todo:
    * Parse commandline arguments
    * Implement learning rate decay

"""

import argparse
import logging
import time

import tensorflow as tf
import numpy as np

import lm
import common_utils
import data_utils

logger = common_utils.get_logger()
logger.setLevel(logging.DEBUG)


def run_epoch(sess, m, data_iter, train_op=tf.no_op()):
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
        if m.opt.is_training and step % 1000 == 0:
            logger.info("{} perplexity: {} wps: {}".format(
                    step, np.exp(costs / (step+1)),
                    num_words / (time.time() - start_time)))
    return np.exp(costs / (step+1))

logger.info('Loading data set...')
vocab = data_utils.Vocabulary.from_vocab_file('data/ptb/preprocess/vocab.txt')
train_iter = data_utils.DataIterator(vocab, 'data/ptb/preprocess/train.jsonl')
valid_iter = data_utils.DataIterator(vocab, 'data/ptb/preprocess/valid.jsonl')

with tf.Session() as sess:
    logger.info('Creating model...')
    init_scale = 0.1
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
        model = lm.LM(lm.ModelOption())
        train_op = lm.train_op(model, model.opt)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
        vmodel = lm.LM(lm.ModelOption(is_training=False))

    logger.debug('Trainable variables:')
    for v in tf.trainable_variables():
        logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
    logger.info('Initializing vairables...')
    sess.run(tf.initialize_all_variables())
    train_ppl = run_epoch(sess, model, train_iter, train_op)
    valid_ppl = run_epoch(sess, vmodel, valid_iter)
    print(train_ppl)
    print(valid_ppl)
