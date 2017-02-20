import tensorflow as tf
""" Recurrent langauge models. This code is adapted from:
`Rafal Jozefowicz's lm <https://github.com/rafaljozefowicz/lm>`_

Todo:
    - Support other types of cells
"""

# \[T]/ PRAISE THE SUN!
#  |_|
#  | |
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

def train_op(model, opt):
    lr = tf.Variable(opt.learning_rate, trainable=False)
    global_step = tf.contrib.framework.get_or_create_global_step()
    optimizer = get_optimizer(lr, opt.optim)
    loss = model.loss * opt.batch_size
    g_v_pairs = optimizer.compute_gradients(loss)
    grads, tvars = [], []
    for g,v in g_v_pairs:
        tvars.append(v)
        grads.append(g)
    clipped_grads, _norm = tf.clip_by_global_norm(
        grads, opt.max_grad_norm)
    g_v_pairs = zip(clipped_grads, tvars)
    train_op = optimizer.apply_gradients(
        g_v_pairs,
        global_step=global_step)
    return train_op, lr

class LM(object):

    def __init__(self, opt, is_training=True):
        self.is_training = is_training
        self.opt = opt
        self._create_input_placeholder(opt)
        self._top_scope = tf.get_variable_scope().name
        with tf.variable_scope(tf.get_variable_scope()):
            self._create_graph(opt)

    def _create_input_placeholder(self, opt):
        """ Setup variables for network input (feeddict) """
        self.x = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps],
                                name='x')
        self.y = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps],
                                name='y')
        self.w = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps],
                                name='w')
        self.seq_len = tf.placeholder(tf.int32, [opt.batch_size],
                                     name='sq_len')

    def _create_graph(self, opt):
        """ Setup model graph starting from the input placeholder.
            The method creates loss and grads for running and training.
        """
        self.loss = self._forward(opt, self.x, self.y, self.w)

    def _forward(self, opt, x, y, w):
        """ Create forward graph. """
        w = tf.to_float(w)
        # Embedding
        self._emb_vars = self._input_emb(opt)
        # Input
        self._inputs = self._input_graph(opt, self._emb_vars, x)
        # RNN
        self._rnn_state, self.initial_state, self.final_state =\
            self._rnn_graph(opt, self._inputs)
        # Modified rnn state
        self._rnn_output, softmax_size = self._modified_rnn_state_graph(
            opt, self._rnn_state)
        # Softmax and loss
        loss, self._all_losses, self._all_logits = self._softmax_loss_graph(
            opt, softmax_size, self._rnn_output, y, w)
        return loss

    def _input_emb(self, opt):
        return tf.get_variable("emb", [opt.vocab_size, opt.emb_size])

    def _input_graph(self, opt, emb_vars, x):
        """ Create input graph before the RNN """
        # [bs, steps, emb_size]
        x = tf.nn.embedding_lookup(emb_vars, x)
        if self.is_training and opt.emb_keep_prob < 1.0:
            x = tf.nn.dropout(x, opt.emb_keep_prob)
        # steps * [bs, emb_size]
        inputs = [tf.squeeze(_x, [1])
                  for _x in tf.split(x, opt.num_steps, 1)]
        # opt.rnn_input_size = opt.emb_size
        return inputs

    def _rnn_graph(self, opt, inputs):
        """ Create RNN graph """
        with tf.variable_scope("rnn") as vs:
            # TODO: Support other types of cells
            cell = tf.contrib.rnn.BasicLSTMCell(opt.state_size)
            if self.is_training and opt.keep_prob < 1.0:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=opt.keep_prob)
            cell_stack = tf.contrib.rnn.MultiRNNCell(
                [cell] * opt.num_layers, state_is_tuple=True)
            initial_state = cell_stack.zero_state(
                opt.batch_size, tf.float32)
            seq_len = None
            if opt.varied_len:
                seq_len = self.seq_len
            outputs, state = tf.contrib.rnn.static_rnn(
                cell_stack, inputs, initial_state=initial_state,
                sequence_length=seq_len)
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, opt.state_size])
        return outputs, initial_state, state

    def _modified_rnn_state_graph(self, opt, rnn_state):
        return rnn_state, opt.state_size

    def _softmax_loss_graph(self, opt, softmax_size, state, y, w):
        """ Create softmax and loss graph """
        softmax_w = tf.get_variable("softmax_w", [opt.vocab_size, softmax_size])
        softmax_b = tf.get_variable("softmax_b", [opt.vocab_size])
        logits = tf.matmul(
            state, softmax_w, transpose_b=True) + softmax_b
        targets = tf.reshape(y, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)
        flat_w = tf.reshape(w, [-1])
        sum_loss = tf.reduce_sum(loss * flat_w)
        mean_loss = sum_loss / (tf.reduce_sum(flat_w) + 1e-12)
        return mean_loss, loss, logits
