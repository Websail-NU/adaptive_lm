import argparse
import logging
import time
import os
import json
import random
import cPickle
random.seed(1234)

import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)

import lm
import common_utils
import data_utils
from exp_utils import *

def get_loss_summary_op(sess, train_lm, train_dm):
    tf.scalar_summary('lm_loss', train_lm.loss)
    tf.scalar_summary('dm_loss', train_dm.loss)
    summary_writer = tf.summary.FileWriter('tf.log', sess.graph)
    return tf.merge_all_summaries(), summary_writer

def get_joint_train_op(train_lm, train_dm, opt_lm, opt_dm):
    with tf.variable_scope('joint_training_ops'):
        loss_lm = train_lm.loss * opt_lm.num_steps
        loss_dm = train_dm.loss * (opt_dm.num_steps * opt_lm.dm_loss_weight)
        joint_loss = loss_lm + loss_dm
        train_vars = tf.trainable_variables()
        grads = tf.gradients(joint_loss, train_vars)
        clipped_grads, _norm = tf.clip_by_global_norm(
            grads, opt_lm.max_grad_norm
        )
        lr = tf.Variable(opt_lm.learning_rate, trainable=False,
                         name="learning_rate")
        global_step = tf.get_variable("global_step", [], tf.float32,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(
            zip(clipped_grads, train_vars),
            global_step=global_step)
        return train_op, lr

def run_joint_epoch(sess, train_lm, train_dm,
                    lm_iter, dm_iter, opt, train_op,
                    summary_writer=None, summary_op=None, global_steps=None):
    start_time = time.time()
    dm_iter.init_batch(train_dm.opt.batch_size)
    cost_lm = 0.0
    cost_dm = 0.0
    num_words_lm = 0
    num_words_dm = 0
    state_lm = []
    state_dm = []
    for c, h in train_lm.initial_state:
        state_lm.append((c.eval(), h.eval()))
    for c, h in train_dm.initial_state:
        state_dm.append((c.eval(), h.eval()))
    for step, (x, y, w, l, seq_len) in enumerate(lm_iter.iterate_epoch(
        opt.batch_size, opt.num_steps)):
        def_x, def_y, def_w, def_l, def_seq_len = dm_iter.next_batch()
        if def_x is None:
            dm_iter.init_batch(train_dm.opt.batch_size)
            def_x, def_y, def_w, def_l, def_seq_len = dm_iter.next_batch()
        feed_dict = {train_lm.x: x, train_lm.y: y,
                     train_lm.w: w, train_lm.seq_len: seq_len,
                     train_dm.x: def_x, train_dm.y: def_y, train_dm.l: def_l,
                     train_dm.w: def_w, train_dm.seq_len: def_seq_len}
        fetches = [train_lm.loss, train_dm.loss, train_op, summary_op]
        for c, h in train_lm.final_state:
            fetches.append(c)
            fetches.append(h)
        for i, (c, h) in enumerate(train_lm.initial_state):
            feed_dict[c], feed_dict[h] = state_lm[i]
        for c, h in train_dm.final_state:
            fetches.append(c)
            fetches.append(h)
        for i, (c, h) in enumerate(train_dm.initial_state):
            feed_dict[c], feed_dict[h] = state_dm[i]
        res = sess.run(fetches, feed_dict)
        state_flat = res[4:4+2*train_lm.opt.num_layers]
        state_lm = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        cost_lm += res[0]
        cost_dm += res[1]
        num_words_lm += np.sum(w)
        num_words_dm += np.sum(def_w)

        if (step + 1) % opt.progress_steps == 0:
            logger.info("-- @{} LM PPL: {}, DM PPL: {}, joint wps: {}".format(
                    step + 1, np.exp(cost_lm / (step + 1)),
                    np.exp(cost_dm / (step + 1)),
                    (num_words_lm + num_words_dm) / (time.time() - start_time)))
        summary_writer.add_summary(res[3], global_steps + step)
    return np.exp(cost_lm / (step+1)), np.exp(cost_dm / (step + 1)), step

def main(opt_lm, opt_dm):
    vocab_lm_path = os.path.join(opt_lm.data_dir, opt_lm.vocab_file)
    train_lm_path = os.path.join(opt_lm.data_dir, opt_lm.train_file)
    valid_lm_path = os.path.join(opt_lm.data_dir, opt_lm.valid_file)
    vocab_dm_path = os.path.join(opt_dm.data_dir, opt_dm.vocab_file)
    train_dm_path = os.path.join(opt_dm.data_dir, opt_dm.train_file)
    logger.info('Loading data set...')
    logger.debug('- Loading vocab LM from {}'.format(vocab_lm_path))
    vocab_lm = data_utils.Vocabulary.from_vocab_file(vocab_lm_path)
    logger.debug('-- LM vocab size: {}'.format(vocab_lm.vocab_size))
    logger.debug('- Loading vocab DM from {}'.format(vocab_dm_path))
    vocab_dm = data_utils.Vocabulary.from_vocab_file(vocab_dm_path)
    logger.debug('-- DM vocab size: {}'.format(vocab_dm.vocab_size))
    logger.debug('- Loading train LM data from {}'.format(train_lm_path))
    train_lm_iter = data_utils.DataIterator(vocab_lm, train_lm_path)
    logger.debug('- Loading valid LM data from {}'.format(valid_lm_path))
    valid_lm_iter = data_utils.DataIterator(vocab_lm, valid_lm_path)
    logger.debug('- Loading train DM data from {}'.format(train_dm_path))
    train_dm_iter = data_utils.DefIterator(vocab_dm, train_dm_path,
                                           l_vocab=vocab_lm)
    logger.info('Loading data completed')

    opt_lm.vocab_size = vocab_lm.vocab_size
    opt_dm.vocab_size = vocab_dm.vocab_size
    opt_dm.num_steps = train_dm_iter._max_seq_len

    init_scale = opt_lm.init_scale
    sess_config =tf.ConfigProto(log_device_placement=False)
    logger.info('Starting TF Session...')
    with tf.Session(config=sess_config) as sess:
        logger.debug(
                '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating shared embedding variables...')
        with tf.variable_scope('shared_emb'):
            shared_emb_vars = lm.sharded_variable(
                'emb', [opt_lm.vocab_size, opt_lm.emb_size], opt_lm.num_shards)
        opt_lm.input_emb_vars = shared_emb_vars
        opt_dm.af_ex_emb_vars = shared_emb_vars
        logger.debug('- Creating training LM...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            # train_lm = lm.LM(opt_lm, create_grads=False)
            train_lm = lm.LM(opt_lm, create_grads=True)
            train_op, lr_var = lm.train_op(train_lm, train_lm.opt)
        logger.debug('- Creating validating LM (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            valid_lm = lm.LM(opt_lm, is_training=False)
        logger.debug('- Creating training DM...')
        with tf.variable_scope('DM', reuse=None, initializer=initializer):
            train_dm = lm.LMwAF(opt_dm, create_grads=False)
        # logger.debug('- Creating training operation...')
        # train_op, lr_var = get_joint_train_op(train_lm, train_dm,
        #                                       opt_lm, opt_dm)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))

        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        state = common_utils.get_initial_training_state()
        state.learning_rate = opt_lm.learning_rate
        state, _ = resume_if_possible(opt_lm, sess, saver, state)
        logger.debug('Creating Summary Writer...')
        summary_op, summary_writer = get_loss_summary_op(sess, train_lm, train_dm)
        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())
        global_steps = 0
        train_dm_ppl = 0
        for epoch in range(state.epoch, opt_lm.max_epochs):
            epoch_time = time.time()
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lr_var, state.learning_rate))
            logger.info("- Learning rate = {}".format(state.learning_rate))
            logger.info("Traning...")
            # train_lm_ppl, train_dm_ppl, steps = run_joint_epoch(
            #     sess, train_lm, train_dm, train_lm_iter,
            #     train_dm_iter, opt_lm, train_op,
            #     summary_writer, summary_op, global_steps)
            train_lm_ppl, lsteps = run_epoch(sess, train_lm, train_lm_iter,
                                             opt_lm, train_op)
            # global_steps += steps
            logger.info("Validating LM...")
            valid_lm_ppl, vsteps = run_epoch(sess, valid_lm,
                                             valid_lm_iter, opt_lm)
            logger.info('DM PPL = {}, Train ppl = {}, Valid ppl = {}'.format(
                        train_dm_ppl, train_lm_ppl, valid_lm_ppl))
            logger.info('----------------------------------')
            logger.info('Post epoch routine...')
            state.epoch = epoch + 1
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
        summary_writer.close()

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--af_mode', type=str, default='gated_state',
                        help='additional feature module type')
    parser.add_argument('--def_dir', type=str,
                        default='data/ptb_defs/preprocess',
                        help='data directory for dictionary corpus')
    parser.add_argument('--dm_loss_weight', type=float, default=0.001,
                        help='weight on DM loss')

    args = parser.parse_args()
    opt_lm = common_utils.Bunch.default_model_options()
    opt_lm.update_from_ns(args)
    opt_dm = common_utils.Bunch.default_model_options()
    opt_dm.update_from_ns(args)
    opt_dm.af_function = 'ex_emb'
    # With GPU, this will slow us down.
    # A proper weights to the loss function is enough to get correct gradients
    opt_dm.varied_len = True
    opt_dm.reset_state = True
    opt_dm.data_dir = opt_dm.def_dir
    logger = common_utils.get_logger(opt_lm.log_file_path)
    if opt_lm.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt_dm.__repr__()))
    main(opt_lm, opt_dm)
    logger.info('Total time: {}s'.format(time.time() - global_time))
