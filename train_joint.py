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

logger = common_utils.get_logger()
logger.setLevel(logging.DEBUG)


tf.set_random_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

def transfer_emb(sess, source, target, index_map):
    s_emb_var = lm.find_trainable_variables(source, "emb")[0]
    t_emb_var = lm.find_trainable_variables(target, "emb")[0]
    s_emb, t_emb = sess.run([s_emb_var, t_emb_var])
    for k in index_map.keys():
        t_emb[index_map[k]] = s_emb[k]
    sess.run(tf.assign(t_emb_var, t_emb))

def run_epoch(sess, m, data_iter, opt, train_op=tf.no_op()):
    """ train the model on the given data. """
    start_time = time.time()
    costs = 0.0
    num_words = 0
    state = []
    for c, h in m.initial_state:
        state.append((c.eval(), h.eval()))
    for step, (x, y, w, l, seq_len) in enumerate(data_iter.iterate_epoch(
        m.opt.batch_size, m.opt.num_steps)):
        fetches = [m.loss, train_op]
        feed_dict = {m.x: x, m.y: y, m.w: w, m.seq_len: seq_len}
        if hasattr(m, 'l'):
            feed_dict[m.l] = l
        if not opt.reset_state:
            for c, h in m.final_state:
                fetches.append(c)
                fetches.append(h)
            for i, (c, h) in enumerate(m.initial_state):
                feed_dict[c], feed_dict[h] = state[i]
        res = sess.run(fetches, feed_dict)
        cost = res[0]
        if not opt.reset_state:
            state_flat = res[2:]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        costs += cost
        num_words += np.sum(w)
        if train_op.name != u'NoOp' and (step + 1) % opt.progress_steps == 0:
            logger.info("-- @{} perplexity: {} wps: {}".format(
                    step + 1, np.exp(costs / (step + 1)),
                    num_words / (time.time() - start_time)))
    return np.exp(costs / (step+1)), step


parser = common_utils.get_common_argparse()
args = parser.parse_args()
opt_lm = common_utils.Bunch.default_model_options()
opt_lm.update_from_ns(args)
opt_dm = common_utils.Bunch.default_model_options()
opt_dm.update_from_ns(args)
opt_dm.af_mode = 'gated_state'
opt_dm.varied_len = True
opt_dm.reset_state = True

vocab_lm_path = "data/ptb/preprocess/vocab.txt"
train_lm_path = "data/ptb/preprocess/train.jsonl"
valid_lm_path = "data/ptb/preprocess/valid.jsonl"
vocab_dm_path = "data/ptb_defs/preprocess/vocab.txt"
train_dm_path = "data/ptb_defs/preprocess/train.jsonl"
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

lm2dm, dm2lm = data_utils.Vocabulary.vocab_index_map(vocab_lm, vocab_dm)

opt_lm.vocab_size = vocab_lm.vocab_size
opt_dm.vocab_size = vocab_dm.vocab_size
init_scale = opt_lm.init_scale

with tf.Session() as sess:
# sess = tf.Session()
# if True:
    logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
    initializer = tf.random_uniform_initializer(-init_scale, init_scale,
                                                seed=1234)
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
    sess.run(tf.initialize_all_variables())
    logger.info('Start training loop:')
    logger.debug('\n' + common_utils.SUN_BRO())

    for epoch in range(3):
        # logger.info("========= Start epoch {} =========".format(epoch+1))
        train_lm_ppl, steps = run_epoch(sess, train_lm, train_lm_iter,
                                        opt_lm, train_lm_op)
        valid_lm_ppl, vsteps = run_epoch(sess, valid_lm, valid_lm_iter, opt_lm)
        # logger.info('Train ppl = {}, Valid ppl = {}'.format(
        #             train_lm_ppl, valid_lm_ppl))
        print('{}\t{}\t{}\t{}'.format(
            epoch+1, "", train_lm_ppl, valid_lm_ppl))

    for epoch in range(3, 13):
        # logger.info("========= Start epoch {} =========".format(epoch+1))
        # logger.info("Traning DM...")
        transfer_emb(sess, "LM", "DM", lm2dm)
        train_dm_ppl, dsteps = run_epoch(sess, train_dm, train_dm_iter,
                                        opt_dm, train_dm_op)
        # logger.info("Traning LM...")
        transfer_emb(sess, "DM", "LM", dm2lm)
        train_lm_ppl, lsteps = run_epoch(sess, train_lm, train_lm_iter,
                                         opt_lm, train_lm_op)
        # logger.info("Validating LM...")
        valid_lm_ppl, vsteps = run_epoch(sess, valid_lm, valid_lm_iter, opt_lm)
        # logger.info('-DM ppl = {}, Train ppl = {}, Valid ppl = {}'.format(
        #             train_dm_ppl, train_lm_ppl, valid_lm_ppl))
        print('{}\t{}\t{}\t{}'.format(
            epoch+1, train_dm_ppl, train_lm_ppl, valid_lm_ppl))


    # XXX: do loop
    # writer = tf.train.SummaryWriter("tf.log", sess.graph)
    # writer.close()

    # logger.info('- Training LM:')
    # train_ppl, steps = run_epoch(sess, train_lm, train_lm_iter,
    #                              opt_lm, train_lm_op)
    # train_ppl, steps = run_epoch(sess, train_dm, train_dm_iter,
    #                              opt_dm, train_dm_op)
