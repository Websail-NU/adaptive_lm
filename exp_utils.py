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
    ckpt_path = os.path.join(opt.output_dir, "{}_model.ckpt".format(prefix))
    state_path = os.path.join(opt.output_dir, "{}_state.json".format(prefix))
    logger.debug('Looking for checkpoint at {}'.format(ckpt_path))
    logger.debug('Looking for state at {}'.format(state_path))
    if os.path.exists(state_path):
        logger.info('Found existing checkpoint, resume state')
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

def resume_many_states(output_dir, sess, saver, states, prefix=["latest"]):
    logger = logging.getLogger("exp")
    count = 0
    for pre in prefix:
        state_path = os.path.join(output_dir, "{}_state.json".format(pre))
        logger.debug('Looking for state at {}'.format(state_path))
        if os.path.exists(state_path):
            with open(state_path) as ifp:
                states[pre].update_from_dict(json.load(ifp))
                count+=1
        else:
            logger.debug('State file: {} is not found.'.format(state_path))
    if count == len(states):
        ckpt_path = os.path.join(output_dir, "{}_model.ckpt".format(prefix[0]))
        logger.debug('Looking for checkpoint at {}'.format(ckpt_path))
        logger.debug('- Restoring model variables...')
        saver.restore(sess, ckpt_path)
        for k in states:
            logger.info("Resumed {} states:\n{}".format(
                k, states[k].__repr__()))
        return states, True
    else:
        logger.info('No state to resume...')
        return states, False


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

def save_model_and_state(sess, saver, state, output_dir, prefix):
    logger = logging.getLogger("exp")
    ckpt_path = os.path.join(output_dir,
                             "{}_model.ckpt".format(prefix))
    state_path = os.path.join(output_dir,
                              "{}_state.json".format(prefix))
    if sess is not None:
        logger.info('- Saving model to {}'.format(ckpt_path))
        saver.save(sess, ckpt_path)
    logger.info('- Saving state to {}'.format(state_path))
    with open(state_path, 'w') as ofp:
        json.dump(vars(state), ofp)

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
        save_model_and_state(sess, saver, state, opt.output_dir, best_prefix)
    else:
        logger.info('- No improvement!')
    done_training = update_lr(opt, state)
    save_model_and_state(sess, saver, state, opt.output_dir, latest_prefix)
    return done_training

def transfer_emb(sess, s_scope, s_name, t_scope, t_name, shortlist, index_map):
    logger = logging.getLogger("exp")
    s_emb_vars = lm.find_variables(s_scope, s_name)
    t_emb_vars = lm.find_variables(t_scope, t_name)
    s_embs = sess.run(s_emb_vars)
    t_embs = sess.run(t_emb_vars)
    logger.info('- Transfering parameters')
    logger.debug('-- From {} ...'.format(
        ', '.join([v.name for v in s_emb_vars])))
    logger.debug('-- To {} ...'.format(
        ', '.join([v.name for v in t_emb_vars])))
    c = 0
    for k in shortlist:
        t_k = index_map[k]
        t_i = 0
        if t_k >= len(t_embs[0]):
            t_i = 1
            t_k = t_k - len(t_embs[0])
        t_embs[t_i][t_k] = s_embs[0][k]
        c = c+1
    logger.debug('-- Completed, total rows: {}'.format(c))
    for i in range(len(t_embs)):
        sess.run(tf.assign(t_emb_vars[i], t_embs[i]))

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
            if opt.sen_independent and data_iter.is_new_sen():
                state = []
                for c, h in m.initial_state:
                    state.append((c.eval(), h.eval()))
            for i, (c, h) in enumerate(m.initial_state):
                feed_dict[c], feed_dict[h] = state[i]
            for c, h in m.final_state:
                fetches.append(c)
                fetches.append(h)
        res = sess.run(fetches, feed_dict)
        cost = res[0]
        if not opt.reset_state:
            state_flat = res[f_state_start:]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        b_num_words = np.sum(w)
        num_words += b_num_words
        costs += cost * b_num_words
        if token_loss is not None:
            for i, (t, p) in enumerate(zip(np.nditer(y), np.nditer(w))):
                if p > 0: token_loss.append((int(t), res[1][i]))
                # token_loss[t,0] += 1
                # token_loss[t,1] += res[1][i]
        if train_op.name != u'NoOp' and (step + 1) % opt.progress_steps == 0:
            logger.info("-- @{} perplexity: {} wps: {}".format(
                    step + 1, np.exp(costs / num_words),
                    num_words / (time.time() - start_time)))
    return np.exp(costs / num_words), step

def load_datasets(opt, iterator_type=None, vocab=None,
                  dataset=['train', 'valid', 'test'], **kwargs):
    logger = logging.getLogger("exp")
    vocab_path = os.path.join(opt.data_dir, opt.vocab_file)
    logger.debug('- Loading vocab from {}'.format(vocab_path))
    out_vocab = data_utils.Vocabulary.from_vocab_file(vocab_path)
    logger.debug('-- Vocab size: {}'.format(out_vocab.vocab_size))
    if vocab is None:
        vocab = out_vocab
    if iterator_type is None:
        iterator_type = data_utils.DataIterator
        if opt.sen_independent:
            iterator_type = data_utils.SentenceIterator
    data = {}
    for d in dataset:
        data_key = '{}_file'.format(d)
        data_path = os.path.join(opt.data_dir, opt.__dict__[data_key])
        logger.debug('- Loading {} data from {}'.format(d, data_path))
        data[d] = iterator_type(vocab, data_path, **kwargs)
        logger.debug('-- Data size: {}'.format(len(data[d]._data)))
    return data, out_vocab
