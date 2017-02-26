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
        vocab_size = self.opt.vocab_size
        if self.opt.input_vocab_size is not None:
            vocab_size = self.opt.input_vocab_size
        emb_var = tf.get_variable(
            "emb", [vocab_size, self.opt.emb_size],
            trainable=self.opt.input_emb_trainable)
        input_emb_var = tf.nn.embedding_lookup(emb_var, input)
        if self.opt.emb_keep_prob < 1.0:
            input_emb_var = tf.nn.dropout(input_emb_var, self.opt.emb_keep_prob)
        # steps * [bs, emb_size]
        input_emb_var = [tf.squeeze(_x, [1])
                          for _x in tf.split(input_emb_var, self.opt.num_steps, 1)]
        return emb_var, input_emb_var

    def unroll_rnn_cell(self, inputs, seq_len, cell, initial_state):
        """ Unroll RNNCell. """
        seq_len = None
        if self.opt.varied_len:
            seq_len = seq_len
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state, sequence_length=seq_len)
        return rnn_outputs, final_state

    def create_output(self, flat_rnn_outputs):
        logits = self.create_output_logit(flat_rnn_outputs)
        probs = tf.nn.softmax(logits)
        return logits, probs

    def create_output_logit(self, features):
        """ Create softmax graph. """
        vocab_size = self.opt.vocab_size
        if self.opt.output_vocab_size is not None:
            vocab_size = self.opt.output_vocab_size
        softmax_size = features.get_shape()[-1]
        softmax_w = tf.get_variable("softmax_w", [vocab_size, softmax_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
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
