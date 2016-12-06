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

import lm
import common_utils
import data_utils

def resume_if_possible(opt, sess, saver, state, prefix="latest"):
    logger = logging.getLogger("exp")
    ckpt_path = os.path.join(opt.output_dir, "latest_model.ckpt")
    state_path = os.path.join(opt.output_dir, "latest_state.json")
    logger.debug('Looking for checkpoint at {}'.format(ckpt_path))
    logger.debug('Looking for state at {}'.format(state_path))
    if os.path.exists(state_path):
        logger.info('Found existing checkpoint, resume training')
        with open(state_path) as ifp:
            logger.debug('- Loading state...')
            state.update_from_dict(json.load(ifp))
        logger.debug('- Restoring model variables...')
        saver.restore(sess, ckpt_path)
        logger.info('Resumed state:\n{}'.format(state.__repr__()))
        return state, True
    else:
        logger.info('No state to resume...')
        return state, False


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
                state.last_imp_epoch + 1, state.epoch + 1))
            state.last_imp_epoch = state.epoch
            state.last_imp_val_ppl = state.val_ppl
            state.imp_wait = 0
        else:
            state.imp_wait = state.imp_wait + 1
            logger.info('- No significant improvement found')
            logger.info('-- last improved since epoch: {}'.format(
                state.last_imp_epoch + 1))
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

def transfer_emb(sess, source, target, shortlist, index_map):
    logger = logging.getLogger("exp")
    s_emb_var = lm.find_trainable_variables(source, "emb")[0]
    t_emb_var = lm.find_trainable_variables(target, "emb")[0]
    s_emb, t_emb = sess.run([s_emb_var, t_emb_var])
    for k in shortlist:
        t_emb[index_map[k]] = s_emb[k]
    sess.run(tf.assign(t_emb_var, t_emb))

def run_epoch(sess, m, data_iter, opt,
              train_op=tf.no_op(), token_loss=None):
    """ train the model on the given data. """
    logger = logging.getLogger("exp")
    start_time = time.time()
    costs = 0.0
    num_words = 0
    state = []
    for c, h in m.initial_state:
        state.append((c.eval(), h.eval()))
    for step, (x, y, w, l, seq_len) in enumerate(data_iter.iterate_epoch(
        m.opt.batch_size, m.opt.num_steps)):
        feed_dict = {m.x: x, m.y: y, m.w: w, m.seq_len: seq_len}
        if hasattr(m, 'l'):
            feed_dict[m.l] = l
        fetches = [m.loss]
        f_state_start = 2
        if token_loss is not None:
            fetches.append(m._all_losses)
            f_state_start += 1
        fetches.append(train_op)
        if not opt.reset_state:
            for c, h in m.final_state:
                fetches.append(c)
                fetches.append(h)
            for i, (c, h) in enumerate(m.initial_state):
                feed_dict[c], feed_dict[h] = state[i]
        res = sess.run(fetches, feed_dict)
        cost = res[0]
        if not opt.reset_state:
            state_flat = res[f_state_start:]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        costs += cost
        num_words += np.sum(w)
        if token_loss is not None:
            for i, t in enumerate(np.nditer(y)):
                token_loss[t,0] += 1
                token_loss[t,1] += res[1][i]
        if train_op.name != u'NoOp' and (step + 1) % opt.progress_steps == 0:
            logger.info("-- @{} perplexity: {} wps: {}".format(
                    step + 1, np.exp(costs / (step + 1)),
                    num_words / (time.time() - start_time)))
    return np.exp(costs / (step+1)), step
