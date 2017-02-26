import argparse
import logging
import time
import os

from adaptive_lm.models.basic_rnnlm import BasicRNNLM
from adaptive_lm.utils import common as common_utils
from adaptive_lm.experiments import lm


def build_test_fn(m):
    nodes = BasicRNNLM.build_full_model_graph(m)
    nodes.fetch.collect = common_utils.LazyBunch(
        token_loss=nodes.losses.token_loss)
    return nodes

training_exp_opt = common_utils.LazyBunch(
    resume = 'latest_lm',
    best = 'best_lm',
    splits = ['train', 'valid'],
    run_split = 'valid',
    model_cls = BasicRNNLM,
    build_train_fn = BasicRNNLM.build_full_model_graph,
    build_test_fn = BasicRNNLM.build_full_model_graph,
    training = True
)

testing_exp_opt = common_utils.LazyBunch(training_exp_opt,
    splits = ['valid'],
    run_split = 'valid',
    build_test_fn = build_test_fn,
    training = False
)

global_time = time.time()
parser = common_utils.get_common_argparse()
args = parser.parse_args()
opt = BasicRNNLM.default_model_options()
opt.update(common_utils.LazyBunch.fromNamespace(args))
logger = common_utils.get_logger(opt.log_file_path)
if opt.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
logger.info('Configurations:\n{}'.format(opt.toPretty()))
_ = lm.run(opt, training_exp_opt, logger)
logger.info('Total time: {}s'.format(time.time() - global_time))
