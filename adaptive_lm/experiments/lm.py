""" Training script

This module creates and trains recurrent neural network language model.

Example:

Todo:
    - Support TensorBoard

"""
import time

import tensorflow as tf
from adaptive_lm.utils.data import load_datasets
from adaptive_lm.utils import common as common_utils
from adaptive_lm.utils import run as run_utils

def _train(opt, exp_opt, sess, saver, dataset, state,
          train_model, valid_model, train_op, lr_var,
          logger):
    logger.info('Start training loop:')
    logger.debug('\n' + common_utils.SUN_BRO())
    for epoch in range(state.epoch, opt.max_epochs):
        epoch_time = time.time()
        state.epoch = epoch
        logger.info("========= Start epoch {} =========".format(epoch+1))
        sess.run(tf.assign(lr_var, state.learning_rate))
        logger.info("- Traning LM with learning rate {}...".format(
            state.learning_rate))
        train_info = run_utils.run_epoch(sess, train_model, dataset['train'],
                                         opt, train_op=train_op,
                                         collect_fn=exp_opt.collect_fn)
        logger.info('- Validating LM...')
        valid_info = run_utils.run_epoch(sess, valid_model, dataset['valid'],
                                         opt, collect_fn=exp_opt.collect_fn)
        logger.info('----------------------------------')
        logger.info('LM post epoch routine...')
        done_training = run_utils.run_post_epoch(
            train_info.ppl, valid_info.ppl, state, opt,
            sess=sess, saver=saver,
            best_prefix=exp_opt.best,
            latest_prefix=exp_opt.resume)
        logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
        if done_training:
            break
    logger.info('Done training at epoch {}'.format(state.epoch + 1))

def _initialize_variables(sess, exp_opt, logger):
    if exp_opt.init_variables is None or len(exp_opt.init_variables) == 0:
        return
    logger.info('Manually initializing variables...')
    for (pattern, value) in exp_opt.init_variables:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, pattern)
        for v in variables:
            logger.debug("- ({}) Assign name: {}, shape: {}".format(
                pattern, v.name, v.get_shape()))
            sess.run(tf.assign(v, value))

def run(opt, exp_opt, logger):
    dataset, vocab = load_datasets(opt, dataset=exp_opt.splits,
                                   iterator_type=exp_opt.iterator_cls)
    opt.vocab_size = vocab.vocab_size
    init_scale = opt.init_scale
    logger.debug('Staring session...')
    sess_config = common_utils.get_tf_sess_config(opt)
    with tf.Session(config=sess_config) as sess:
        train_model, test_model, train_op, lr_var = run_utils.create_model(
            opt, exp_opt)
        sess.run(tf.global_variables_initializer())
        _initialize_variables(sess, exp_opt, logger)
        saver = tf.train.Saver()
        if exp_opt.training:
            states, success = run_utils.load_model_and_states(
                opt.experiment_dir, sess, saver, [exp_opt.resume])
            state = states[exp_opt.resume]
            if not success:
                state.learning_rate = opt.learning_rate
            _train(opt, exp_opt, sess, saver, dataset, state,
                   train_model, test_model, train_op, lr_var, logger)
        _, _ = run_utils.load_model_and_states(
            opt.experiment_dir, sess, saver, [exp_opt.best])
        logger.info('Running LM...')
        info = run_utils.run_epoch(sess, test_model, dataset[exp_opt.run_split],
                                   opt, collect_fn=exp_opt.collect_fn)
        return info
