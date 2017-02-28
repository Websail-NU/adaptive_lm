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
        steps = int(input_emb_var.get_shape()[1])
        input_emb_var = [tf.squeeze(_x, [1])
                         for _x in tf.split(input_emb_var, steps, 1)]
        return emb_var, input_emb_var

    def unroll_rnn_cell(self, inputs, seq_len, cell, initial_state):
        """ Unroll RNNCell. """
        seq_len = None
        if self.opt.varied_len:
            seq_len = seq_len
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state, sequence_length=seq_len)
        return rnn_outputs, final_state

    def _flat_rnn_outputs(self, rnn_outputs):
        state_size = int(rnn_outputs[0].get_shape()[-1])
        return tf.reshape(tf.concat(rnn_outputs, 1),
                          [-1, state_size]), state_size

    def create_output(self, rnn_outputs, logit_weights=None):
        if isinstance(rnn_outputs, list):
            flat_output, _ = self._flat_rnn_outputs(rnn_outputs)
        else:
            flat_output = rnn_outputs
        logits = self.create_output_logit(flat_output, logit_weights)
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
                'enc_input_emb_trainable', self.opt.input_emb_trainable)
            enc_emb = tf.get_variable(
                "enc_emb", [vocab_size, emb_size], trainable=trainable)
        encoder = tf.nn.embedding_lookup(enc_emb, enc_inputs)
        if self.opt.emb_keep_prob < 1.0:
            encoder = tf.nn.dropout(encoder, self.opt.emb_keep_prob)
        steps = int(encoder.get_shape()[1])
        # rearrange to fix rnn output
        encoder = [tf.squeeze(_x, [1]) for _x in tf.split(encoder, steps, 1)]
        return encoder

    def create_enc_dec_mixer(self, enc_outputs, dec_outputs):
        """ Combine encoder and decoder into a feature for output
            Args:
                enc_outputs: A list of tensors where each tensor is a encoder output at a time step.
                i.e. [Tensor(batch, hidden),...,Tensor(batch, hidden)]
                dec_outputs: A list of tensors where each tensor is a decoder output at a time step.
                i.e. [Tensor(batch, hidden),...,Tensor(batch, hidden)]
            Returns:
                (feature tensor(batch*steps, hidden), hidden size)
        """
        flat_enc, enc_size = self._flat_rnn_outputs(enc_outputs)
        flat_dec, dec_size = self._flat_rnn_outputs(dec_outputs)
        enc_dec = tf.concat([flat_enc, flat_dec], 1)
        full_size = enc_size + dec_size
        zr_w = tf.get_variable("att_zr_w", [full_size, full_size])
        zr_b = tf.get_variable("att_zr_b", [full_size])
        zr = tf.sigmoid(tf.matmul(enc_dec, zr_w) + zr_b)
        z = tf.slice(zr, [0, 0], [-1, dec_size],
                     name="att_z_gate")
        r = tf.slice(zr, [0, dec_size], [-1, -1],
                     name="att_r_gate")
        att_flat_enc = tf.multiply(flat_enc, r)
        att_enc_dec = tf.concat([att_flat_enc, flat_dec], 1)
        h_w = tf.get_variable("att_h_w",
                              [full_size, dec_size])
        h_b = tf.get_variable("att_h_b", [dec_size])
        h = tf.tanh(tf.matmul(att_enc_dec, h_w) + h_b)
        outputs = tf.multiply((1-z), flat_dec) + tf.multiply(z, h)
        return outputs, dec_size
