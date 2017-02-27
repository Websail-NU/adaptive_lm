"""
TODO:
    - Fix class hiearachy
"""

import tensorflow as tf

class BasicRNNHelper(object):

    def __init__(self, opt):
        self.opt = opt

    def create_input_placeholder(self):
        """ Setup variables for network input (feeddict) """
        inputs = tf.placeholder(tf.int32,
                                [self.opt.batch_size, self.opt.num_steps],
                                name='inputs')
        seq_len = tf.placeholder(tf.int32,
                                 [self.opt.batch_size],
                                 name='seq_len')
        return inputs, seq_len

    def create_input_lookup(self, input):
        """ Create input embedding lookup """
        vocab_size = self.opt.get('input_vocab_size', self.opt.vocab_size)
        emb_var = tf.get_variable(
            "emb", [vocab_size, self.opt.emb_size],
            trainable=self.opt.input_emb_trainable)
        input_emb_var = tf.nn.embedding_lookup(emb_var, input)
        if self.opt.emb_keep_prob < 1.0:
            input_emb_var = tf.nn.dropout(input_emb_var, self.opt.emb_keep_prob)
        # steps * [bs, emb_size]
        input_emb_var = [tf.squeeze(_x, [1])
                         for _x in tf.split(
                             input_emb_var, self.opt.num_steps, 1)]
        return emb_var, input_emb_var

    def unroll_rnn_cell(self, inputs, seq_len, cell, initial_state):
        """ Unroll RNNCell. """
        seq_len = None
        if self.opt.varied_len:
            seq_len = seq_len
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state, sequence_length=seq_len)
        return rnn_outputs, final_state

    def create_output(self, flat_rnn_outputs, logit_weights=None):
        logits = self.create_output_logit(flat_rnn_outputs, logit_weights)
        probs = tf.nn.softmax(logits)
        return logits, probs

    def create_output_logit(self, features, logit_weights):
        """ Create softmax graph. """
        if self.opt.get('tie_input_output_emb', False):
            softmax_w = logit_weights
        else:
            softmax_size = features.get_shape()[-1]
            vocab_size = self.opt.get('output_vocab_size' ,self.opt.vocab_size)
            softmax_w = tf.get_variable("softmax_w", [vocab_size, softmax_size])
        softmax_b = tf.get_variable("softmax_b", softmax_w.get_shape()[0])
        logits =tf.matmul(features, softmax_w, transpose_b=True) + softmax_b
        return logits

    def create_target_placeholder(self):
        """ create target placeholders """
        targets = tf.placeholder(tf.int32,
                                 [self.opt.batch_size, self.opt.num_steps],
                                name='targets')
        weights = tf.placeholder(tf.float32,
                                 [self.opt.batch_size, self.opt.num_steps],
                                name='weights')
        return targets, weights

    def create_xent_loss(self, logits, targets, weights):
        """ create cross entropy loss """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)
        sum_loss = tf.reduce_sum(loss * weights)
        mean_loss = sum_loss / (tf.reduce_sum(weights) + 1e-12)
        return loss, mean_loss

class EmbDecoderRNNHelper(BasicRNNHelper):

    def __init__(self, opt):
        self.opt = opt

    def create_enc_input_placeholder(self):
        enc_inputs = tf.placeholder(
            tf.int32, [self.opt.batch_size, self.opt.num_steps],
            name='enc_inputs')
        return enc_inputs

    def create_encoder(self, enc_inputs, emb_var=None):
        if self.opt.get('tie_input_enc_emb', False):
            enc_emb = emb_var
        else:
            vocab_size = self.opt.get('enc_vocab_size', self.opt.vocab_size)
            emb_size = self.opt.get('enc_emb_size', self.opt.emb_size)
            trainable = self.opt.get(
                'enc_input_emb_trainable', self.input_emb_trainable)
            enc_emb = tf.get_variable(
                "enc_emb", [vocab_size, emb_size], trainable=trainable)
        encoder = tf.nn.embedding_lookup(enc_emb, input)
        if self.opt.emb_keep_prob < 1.0:
            encoder = tf.nn.dropout(encoder, self.opt.emb_keep_prob)
        # steps * [bs, emb_size]
        encoder = [tf.squeeze(_x, [1])
                         for _x in tf.split(
                             encoder, self.opt.num_steps, 1)]
        return encoder
