import argparse
import logging
import time
import os
import cPickle
import numpy as np

from adaptive_lm.models.rnnlm_helper import EmbDecoderRNNHelper
from adaptive_lm.models.basic_rnnlm import DecoderRNNLM
from adaptive_lm.utils import common as common_utils
from adaptive_lm.experiments import lm
from adaptive_lm.utils.data import SenLabelIterator

training_exp_opt = common_utils.LazyBunch(
    resume = 'latest_lm',
    best = 'best_lm',
    splits = ['train', 'valid'],
    run_split = 'valid',
    iterator_cls = SenLabelIterator,
    model_scope = 'DM',
    model_helper_cls = EmbDecoderRNNHelper,
    model_cls = DecoderRNNLM,
    build_train_fn = DecoderRNNLM.build_full_model_graph,
    build_test_fn = DecoderRNNLM.build_full_model_graph,
    init_variables = [],
    training = True,
)

if __name__ == '__main__':
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--emb_pickle_file', type=str,
                        default='emb.cpickle',
                        help='embedding cpickled file in data_dir')
    parser.add_argument('--tie_input_enc_emb', dest='tie_input_enc_emb',
                        action='store_true')
    parser.set_defaults(data_dir='data/common_defs_v1.2/wordnet/preprocess/',
                        state_size=300,
                        emb_size=300,
                        num_layers=2,
                        emb_keep_prob=0.75,
                        keep_prob=0.50,
                        sen_independent=True)
    args = parser.parse_args()
    opt = common_utils.update_opt(DecoderRNNLM.default_model_options(), parser)
    opt.input_emb_trainable = False
    common_utils.ensure_dir(opt.experiment_dir)
    if opt.save_config_file is not None:
        common_utils.save_config_file(opt)
    logger = common_utils.get_logger(os.path.join(opt.experiment_dir, opt.log_file))
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.toPretty()))
    emb_path = os.path.join(opt.data_dir, opt.emb_pickle_file)
    logger.info('Loading embeddings from {}'.format(emb_path))
    with open(emb_path) as ifp:
        emb_values = cPickle.load(ifp)
        training_exp_opt.init_variables.append(
            ('{}/.*{}'.format('DM', 'emb'), emb_values))
    info = lm.run(opt, training_exp_opt, logger)
    logger.info('Perplexity: {}, Num tokens: {}'.format(
        np.exp(info.cost / info.num_words), info.num_words))
    logger.info('Total time: {}s'.format(time.time() - global_time))
