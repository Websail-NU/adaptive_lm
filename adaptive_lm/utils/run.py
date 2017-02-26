""" Experiment Utilities

This module contains a collection of functions to help running experiemnts

Example:

Todo:
    - Better way to set global logger?

"""

import logging
import time
import os
import json

import numpy as np
import tensorflow as tf

import data
from common import LazyBunch
from common import get_initial_training_state

def load_model_and_states(experiment_dir, sess, saver, prefix=["latest"]):
    logger = logging.getLogger("exp")
    states = {}
    for p in prefix:
        states[p] = get_initial_training_state()
    count = 0
    for pre in prefix:
        state_path = os.path.join(experiment_dir, "{}_state.json".format(pre))
        logger.debug('Looking for state at {}'.format(state_path))
        if os.path.exists(state_path):
            with open(state_path) as ifp:
                states[pre] = LazyBunch.fromDict(json.load(ifp))
                count+=1
        else:
            logger.debug('State file: {} is not found.'.format(state_path))
    if count == len(states):
        ckpt_path = os.path.join(experiment_dir, "{}_model.ckpt".format(prefix[0]))
        logger.debug('Looking for checkpoint at {}'.format(ckpt_path))
        logger.debug('- Restoring model variables...')
        saver.restore(sess, ckpt_path)
        for k in states:
            logger.info("Resumed {} states:\n{}".format(
                k, states[k].toPretty()))
        return states, True
    else:
        logger.info('No state to resume...')
        return states, False

def save_model_and_state(sess, saver, state, experiment_dir, prefix):
    logger = logging.getLogger("exp")
    ckpt_path = os.path.join(experiment_dir,
                             "{}_model.ckpt".format(prefix))
    state_path = os.path.join(experiment_dir,
                              "{}_state.json".format(prefix))
    if sess is not None:
        logger.debug('- Saving model to {}'.format(ckpt_path))
        saver.save(sess, ckpt_path)
    logger.debug('- Saving state to {}'.format(state_path))
    with open(state_path, 'w') as ofp:
        ofp.write(state.toPrettyJSON())

def update_lr(opt, state):
    logger = logging.getLogger("exp")
    logger.info('Updating learning rate...')
    old_lr = state.learning_rate
    new_lr = old_lr
    if opt.lr_decay_every > 0 and state.epoch % opt.lr_decay_every == 0:
        new_lr = old_lr * opt.lr_decay_factor
        logger.info('- Scheduled learning rate decay')
    elif opt.lr_decay_imp > 0:
        if state.val_ppl / state.last_imp_val_ppl < opt.lr_decay_imp:
            logger.info('- Significant improvement found')
            logger.info('-- epoch: {} -> {}'.format(
                state.last_imp_epoch, state.epoch))
            state.last_imp_epoch = state.epoch
            state.last_imp_val_ppl = state.val_ppl
            state.imp_wait = 0
        else:
            state.imp_wait = state.imp_wait + 1
            logger.info('- No significant improvement found')
            logger.info('-- last improved since epoch: {}'.format(
                state.last_imp_epoch))
            logger.info('-- waiting for {}/{} epochs'.format(
                state.imp_wait, opt.lr_decay_wait))
            if state.imp_wait >= opt.lr_decay_wait:
                new_lr = old_lr * opt.lr_decay_factor
                state.imp_wait = 0
                if opt.lr_decay_factor == 1.0:
                    logger.info('- Learning rate is constant!')
                    return True
                logger.info('-- decay learning rate')
    logger.info('- Learning rate: {} -> {}'.format(old_lr, new_lr))
    state.learning_rate = new_lr
    if new_lr < opt.min_learning_rate:
        logger.info('- Learning rate reaches mininum ({})!'.format(
            opt.min_learning_rate))
        return True
    return False

def feed_state(feed_dict, state_vars, state_vals):
    for i, (c, h) in enumerate(state_vars):
        feed_dict[c] = state_vals[i].c
        feed_dict[h] = state_vals[i].h
    return feed_dict

def map_feeddict(batch, model_feed):
    feed_dict = {}
    for k in model_feed.keys():
        feed_dict[model_feed[k]] = batch[k]
    return feed_dict

def scheduled_report(opt, info):
    logger = logging.getLogger("exp")
    if (info.step + 1) % opt.progress_steps == 0:
        logger.info("-- @{} perplexity: {} wps: {}".format(
                info.step + 1, np.exp(info.cost / info.num_words),
                info.num_words / (time.time() - info.start_time)))

def run_epoch(sess, m, data, opt, train_op=tf.no_op(), collect_fn=None):
    """ train the model on the given data. """
    logger = logging.getLogger("exp")
    info = LazyBunch(start_time = time.time(), cost = 0.0,
                     num_words = 0, step=0, collect=[])
    b, r = opt.batch_size, opt.num_steps
    zero_state = sess.run(m.init_state)
    state = zero_state
    fetch = LazyBunch(m.fetch, _=train_op)
    for info.step, batch in enumerate(data.iterate_epoch(b, r)):
        feed_dict = map_feeddict(batch, m.feed)
        if opt.sen_independent and batch.new:
            state = zero_state
        feed_dict = feed_state(feed_dict, m.init_state, state)
        result = sess.run(fetch, feed_dict)
        state = result.final_state
        info.cost += result.eval_loss * batch.total
        info.num_words += batch.total
        if result.collect is not None:
            if collect_fn is None:
                info.collect.append(result.collect)
            else:
                collect_fn(result.collect)
        scheduled_report(opt, info)
    info.end_time = time.time()
    info.ppl = np.exp(info.cost / info.num_words)
    return info

def run_post_epoch(new_train_ppl, new_valid_ppl,
                   state, opt, sess=None, saver=None,
                   best_prefix="best", latest_prefix="latest"):
    logger = logging.getLogger("exp")
    logger.info('Train ppl = {}, Valid ppl = {}'.format(
        new_train_ppl, new_valid_ppl))
    state.epoch = state.epoch + 1
    state.val_ppl = new_valid_ppl
    if new_valid_ppl < state.best_val_ppl:
        logger.info('- Best PPL: {} -> {}'.format(
            state.best_val_ppl, new_valid_ppl))
        logger.info('- Epoch: {} -> {}'.format(
            state.best_epoch, state.epoch))
        state.best_val_ppl = new_valid_ppl
        state.best_epoch = state.epoch
        save_model_and_state(sess, saver, state, opt.experiment_dir, best_prefix)
    else:
        logger.info('- No improvement!')
    done_training = update_lr(opt, state)
    save_model_and_state(sess, saver, state, opt.experiment_dir, latest_prefix)
    return done_training

def get_optimizer(lr_var, optim):
    optimizer = None
    if optim == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(lr_var)
    elif optim == "adam":
        optimizer = tf.train.AdamOptimizer(lr_var)
    else:
        import warnings
        warnings.warn('Unsupported optimizer. Use sgd as substitute')
        optimizer = tf.train.GradientDescentOptimizer(lr_var)
    return optimizer

def train_op(loss, opt):
    lr = tf.Variable(opt.learning_rate, trainable=False)
    global_step = tf.contrib.framework.get_or_create_global_step()
    optimizer = get_optimizer(lr, opt.optim)
    loss = loss * opt.batch_size
    g_v_pairs = optimizer.compute_gradients(loss)
    grads, tvars = [], []
    for g,v in g_v_pairs:
        if g is None:
            continue
        tvars.append(v)
        if "embedding_lookup" in g.name:
            assert isinstance(g, tf.IndexedSlices)
            grads.append(tf.IndexedSlices(g.values * opt.batch_size,
                                          g.indices, g.dense_shape))
        else:
            grads.append(g)
    clipped_grads, _norm = tf.clip_by_global_norm(
        grads, opt.max_grad_norm)
    g_v_pairs = zip(clipped_grads, tvars)
    train_op = optimizer.apply_gradients(
        g_v_pairs,
        global_step=global_step)
    return train_op, lr

def create_model(opt, exp_opt):
    logger = logging.getLogger("exp")
    init_scale = opt.init_scale
    logger.debug(
        '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    logger.debug('- Creating training model...')
    with tf.variable_scope(exp_opt.model_scope, reuse=None, initializer=initializer):
        train_model = exp_opt.build_train_fn(exp_opt.model_cls(
            opt, helper=exp_opt.model_helper_cls(opt)))
        optim_op, lr_var = train_op(
            train_model.losses.loss, opt)
    logger.debug('- Creating testing model (reuse params)...')
    with tf.variable_scope(exp_opt.model_scope, reuse=True, initializer=initializer):
        test_opt = LazyBunch(opt, keep_prob=1.0, emb_keep_prob=1.0)
        test_model = exp_opt.build_test_fn(exp_opt.model_cls(
            test_opt, helper=exp_opt.model_helper_cls(test_opt)))
    logger.debug('Trainable variables:')
    for v in tf.trainable_variables():
        logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
    return train_model, test_model, optim_op, lr_var
