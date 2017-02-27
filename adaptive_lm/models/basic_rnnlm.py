import tensorflow as tf
import rnnlm
from adaptive_lm.utils.common import LazyBunch
from rnnlm_helper import BasicRNNHelper

class BasicRNNLM(rnnlm.RNNLM):
    """A basic RNNLM."""

    def __init__(self, opt, cell=None, helper=None):
        """Initialize BasicDecoder.

        Args:
            opt: a LazyBunch object.
            cell: (Optional) an instance of RNNCell. Default is BasicLSTMCell
            help: (Optional) an RNNHelper
        """
        self._opt = LazyBunch(opt)
        self._cell = cell
        if cell is None:
            self._cell = rnnlm.get_rnn_cell(opt.state_size, opt.num_layers,
                                            opt.cell_type, opt.keep_prob)
        self.helper = helper
        if self.helper is None:
            self.helper = BasicRNNHelper(opt)
        helper._model = self

    def batch_size(self):
        return self._opt.batch_size

    def num_steps(self):
        return self._opt.num_steps

    def initialize(self):
        self._input, self._seq_len = self.helper.create_input_placeholder()
        self._emb, self._input_emb = self.helper.create_input_lookup(
            self._input)
        self._initial_state = self._cell.zero_state(
            self._opt.batch_size, tf.float32)
        inputs = LazyBunch(inputs=self._input, seq_len=self._seq_len)
        return inputs, self._initial_state

    def forward(self):
        self._emb, self._input_emb = self.helper.create_input_lookup(
            self._input)
        self._rnn_output, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        flat_output = tf.reshape(tf.concat(self._rnn_output, 1),
                                 [-1, self._opt.state_size])
        self._logit, self._prob = self.helper.create_output(
            flat_output, self._emb)
        self._prob = self._reshape_batch_time(self._prob)
        outputs = LazyBunch(rnn_outputs=self._rnn_output,
                            distributions=self._prob)
        return outputs, self._final_state

    def loss(self):
        self._target, self._weight = self.helper.create_target_placeholder()
        targets = tf.reshape(self._target, [-1])
        weights = tf.reshape(self._weight, [-1])
        self._token_loss, self._loss = self.helper.create_xent_loss(
            self._logit, targets, weights)
        self._token_loss = self._reshape_batch_time(self._token_loss, True)
        target_holder = LazyBunch(targets=self._target, weights=self._weight)
        losses = LazyBunch(token_loss=self._token_loss, loss=self._loss)
        return target_holder, losses

    @staticmethod
    def build_full_model_graph(m):
        inputs, init_state = m.initialize()
        outputs, final_state = m.forward()
        targets, losses = m.loss()
        feed = LazyBunch(
            inputs=inputs.inputs,
            targets=targets.targets,
            weights=targets.weights,
            lengths=inputs.seq_len)
        fetch = LazyBunch(
            final_state=final_state,
            eval_loss=losses.loss
        )
        return LazyBunch(
            inputs=inputs, init_state=init_state, outputs=outputs,
            final_state=final_state, targets=targets, losses=losses,
            feed=feed, fetch=fetch)

    def _reshape_batch_time(self, node, squeeze=False):
        shape = [self._opt.batch_size, self._opt.num_steps, -1]
        if squeeze:
            shape = [self._opt.batch_size, self._opt.num_steps]
        return tf.reshape(
            node, shape)

    @staticmethod
    def default_model_options():
        return LazyBunch(
            batch_size=32,
            num_steps=10,
            num_layers=1,
            varied_len=False,
            emb_keep_prob=0.9,
            keep_prob=0.75,
            vocab_size=10000,
            emb_size=100,
            state_size=100,
            input_emb_trainable=True
        )

class DecoderRNNLM(BasicRNNLM):
    """A decoder RNNLM."""

    def __init__(self, opt, cell=None, helper=None):
        """Initialize BasicDecoder.

        Args:
            opt: a LazyBunch object.
            cell: (Optional) an instance of RNNCell. Default is BasicLSTMCell
            help: (Optional) an DecoderHelper
        """
        if helper is None:
            helper = EmbDecoderRNNHelper(opt)
        super(DecoderRNNLM, self).__init__(opt, cell, helper)

    def initialize(self):
        inputs, self._initial_state = super(DecoderRNNLM, self).initialize()
        self._enc_input = helper.create_enc_input_placeholder()
        inputs.enc_inputs = self._enc_input
        return inputs, self._initial_state

    def forward(self):
        self._rnn_output, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        flat_output = tf.reshape(tf.concat(self._rnn_output, 1),
                                 [-1, self._opt.state_size])
        self._enc_output = self.helper.create_encoder(
            self._enc_input, self._emb)
        # Create attention (combine decoder states and encoder's output)
        self._logit, self._prob = self.helper.create_output(
            flat_output, self._emb)
        self._prob = self._reshape_batch_time(self._prob)
        outputs = LazyBunch(rnn_outputs=self._rnn_output,
                            distributions=self._prob)
        return outputs, self._final_state

    @staticmethod
    def build_full_model_graph(m):
        nodes = BasicRNNLM.build_full_model_graph(m)
        nodes.feed.enc_inputs = nodes.inputs.enc_inputs
