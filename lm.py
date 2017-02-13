import tensorflow as tf
""" Recurrent langauge models. This code is adapted from:
`Rafal Jozefowicz's lm <https://github.com/rafaljozefowicz/lm>`_

Todo:
    - Support other types of cells
    - Refactor LMwAF._extract_features
    - Add char-CNN
"""

# \[T]/ PRAISE THE SUN!
#  |_|
#  | |

def find_variables(top_scope, key):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                             "{}/.*{}.*".format(top_scope, key))

def find_trainable_variables(top_scope, key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             "{}/.*{}.*".format(top_scope, key))

def sharded_variable(name, shape, num_shards, trainable=True,
                     dtype=tf.float32, initializer=None):
    # The final size of the sharded variable may be larger than requested.
    # This should be fine for embeddings.
    shard_size = int((shape[0] + num_shards - 1) / num_shards)
    if initializer is None:
        initializer = tf.uniform_unit_scaling_initializer(dtype=dtype)
    if isinstance(initializer, tf.Tensor):
        return [tf.get_variable(name+"_0",
                                initializer=initializer,
                                dtype=dtype, trainable=trainable)]
    return [tf.get_variable(name + "_%d" % i,
                            [shard_size, shape[1]],
                            initializer=initializer,
                            dtype=dtype)
            for i in range(num_shards)]

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
    global_step = tf.get_variable("global_step", [], tf.float32,
                                  initializer=tf.zeros_initializer,
                                  trainable=False)
    optimizer = get_optimizer(lr, opt.optim)
    train_op = optimizer.apply_gradients(
        zip(model.grads, model.vars),
        global_step=global_step)
    return train_op, lr

class LM(object):

    def __init__(self, opt, is_training=True, create_grads=True):
        self.is_training = is_training
        self.create_grads = create_grads
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
        if self.is_training and self.create_grads:
            self.grads, self.vars = self._backward(opt, self.loss)

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
        """ Create embedding variable or reuse set at opt.input_emb_vars
        """
        emb_vars = None
        if hasattr(opt, 'input_emb_vars'):
            emb_vars = opt.input_emb_vars
        elif hasattr(opt, 'input_emb_init'):
            emb_vars = sharded_variable(
                "emb", [1,1], 1, initializer=opt.input_emb_init,
                trainable=opt.input_emb_trainable)
        else:
            emb_vars = sharded_variable(
                "emb", [opt.vocab_size, opt.emb_size], opt.num_shards,
                trainable=opt.input_emb_trainable)
        return emb_vars

    def _input_graph(self, opt, emb_vars, x):
        """ Create input graph before the RNN """
        # [bs, steps, emb_size]
        x = tf.nn.embedding_lookup(emb_vars, x)
        if self.is_training and opt.emb_keep_prob < 1.0:
            x = tf.nn.dropout(x, opt.emb_keep_prob)
        # steps * [bs, emb_size]
        inputs = [tf.squeeze(_x, [1])
                  for _x in tf.split(1, opt.num_steps, x)]
        # opt.rnn_input_size = opt.emb_size
        return inputs

    def _rnn_graph(self, opt, inputs):
        """ Create RNN graph """
        with tf.variable_scope("rnn") as vs:
            # TODO: Support other types of cells
            cell = tf.nn.rnn_cell.BasicLSTMCell(opt.state_size)
            if self.is_training and opt.keep_prob < 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=opt.keep_prob)
            cell_stack = tf.nn.rnn_cell.MultiRNNCell(
                [cell] * opt.num_layers, state_is_tuple=True)
            initial_state = cell_stack.zero_state(
                opt.batch_size, tf.float32)
            seq_len = None
            if opt.varied_len:
                seq_len = self.seq_len
            outputs, state = tf.nn.rnn(
                cell_stack, inputs, initial_state=initial_state,
                sequence_length=seq_len)
        outputs = tf.reshape(tf.concat(1, outputs), [-1, opt.state_size])
        return outputs, initial_state, state

    def _modified_rnn_state_graph(self, opt, rnn_state):
        return rnn_state, opt.state_size

    def _softmax_w(self, opt, softmax_size):
        softmax_w = None
        if hasattr(opt, 'softmax_w_vars'):
            softmax_w = opt.softmax_w_vars
        else:
            softmax_w = sharded_variable(
                "softmax_w", [opt.vocab_size, softmax_size], opt.num_shards)
        return softmax_w

    def _logit_mask(self, opt):
        mask = opt.logit_mask
        return tf.reshape(tf.constant((mask - 1) * 100000,
                                      name="logit_mask",
                                      dtype=tf.float32),
                          [1, -1])

    def _softmax_loss_graph(self, opt, softmax_size, state, y, w):
        """ Create softmax and loss graph """
        softmax_w = self._softmax_w(opt, softmax_size)
        softmax_b = tf.get_variable("softmax_b", [opt.vocab_size])
        logits = None
        # only sample when training
        if opt.num_softmax_sampled == 0 or not self.is_training:
            with tf.variable_scope("softmax_w"):
                full_softmax_w = tf.reshape(
                    tf.concat(1, softmax_w), [-1, softmax_size])
                # full_softmax_w = full_softmax_w[:opt.vocab_size, :]
            logits = tf.matmul(
                state, full_softmax_w, transpose_b=True) + softmax_b
            if hasattr(opt, 'logit_mask'):
                logits = logits + self._logit_mask(opt)
            targets = tf.reshape(y, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, targets)
        else:
            targets = tf.reshape(y, [-1, 1])
            loss = tf.nn.sampled_softmax_loss(
                softmax_w, softmax_b, tf.to_float(state),
                targets, opt.num_softmax_sampled, opt.vocab_size)
        # mean_loss = tf.reduce_mean(loss * tf.reshape(w, [-1]))
        flat_w = tf.reshape(w, [-1])
        sum_loss = tf.reduce_sum(loss * flat_w)
        mean_loss = sum_loss / (tf.reduce_sum(flat_w) + 1e-12)
        return mean_loss, loss, logits

    def _backward(self, opt, loss):
        """ Create gradient graph (including gradient clips).
            Return clipped gradients and trainable variables
        """
        loss = loss * opt.num_steps
        emb_vars, rnn_vars, softmax_vars, other_vars = self._get_variables(opt)
        all_vars = emb_vars + rnn_vars + softmax_vars + other_vars
        grads = tf.gradients(loss, all_vars)
        orig_grads = grads[:]
        # getting embedding gradients from shards
        emb_grads = grads[:len(emb_vars)]
        grads = grads[len(emb_vars):]
        for i in range(len(emb_grads)):
            assert isinstance(emb_grads[i], tf.IndexedSlices)
            emb_grads[i] = tf.IndexedSlices(
                emb_grads[i].values * opt.batch_size, emb_grads[i].indices,
                emb_grads[i].dense_shape)
        # getting rnn gradients for cliping
        rnn_grads = grads[:len(rnn_vars)]
        # rnn_grads, rnn_norm = tf.clip_by_global_norm(rnn_grads, opt.max_grad_norm)
        # the rest of the gradients
        rest_grads = grads[len(rnn_vars):]
        all_grads = emb_grads + rnn_grads + rest_grads
        clipped_grads, _norm = tf.clip_by_global_norm(
            all_grads, opt.max_grad_norm)
        assert len(clipped_grads) == len(orig_grads)
        return clipped_grads, all_vars

    def _get_variables(self, opt):
        emb_vars = None
        if hasattr(opt, 'input_emb_vars'):
            emb_vars = opt.input_emb_vars
        else:
            emb_vars = find_trainable_variables(self._top_scope, "emb")
        rnn_vars = find_trainable_variables(self._top_scope, "rnn")
        softmax_vars = find_trainable_variables(self._top_scope, "softmax")
        other_vars = self._get_additional_variables()
        return emb_vars, rnn_vars, softmax_vars, other_vars

    def _get_additional_variables(self):
        """ Placeholder method """
        return []

class LMwAF(LM):

    def _create_input_placeholder(self, opt):
        super(LMwAF, self)._create_input_placeholder(opt)
        self.l = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps],
                                name='l')

    def _input_graph(self, opt, emb_vars, x):
        """ Create input graph before the RNN """
        # [bs, steps, emb_size]
        x = tf.nn.embedding_lookup(emb_vars, x)
        if self.is_training and opt.emb_keep_prob < 1.0:
            x = tf.nn.dropout(x, opt.emb_keep_prob)
        # [bs, steps, af_size]
        self._af = self._extract_features(opt, self.l)
        if opt.af_mode == 'concat_input':
            with tf.variable_scope("afeatures"):
                x = tf.concat(2, [self._af, x])
        # steps * [bs, emb_size + af_size]
        inputs = [tf.squeeze(_x, [1])
                  for _x in tf.split(1, opt.num_steps, x)]
        return inputs

    def _modified_rnn_state_graph(self, opt, rnn_state):
        if opt.af_mode == 'concat_input':
            return super(LMwAF, self)._modified_rnn_state_graph(opt, rnn_state)
        with tf.variable_scope("afeatures"):
            l = tf.reshape(self._af, [-1, self._af_size])
            outputs = tf.concat(1, [l, rnn_state])
            full_size = self._af_size + opt.state_size
            if opt.af_mode == 'concat_state':
                proj_w = tf.get_variable("af_h_proj_w",
                                         [full_size, opt.state_size])
                proj_b = tf.get_variable("af_h_proj_b", [opt.state_size])
                outputs = tf.tanh(tf.matmul(outputs, proj_w) + proj_b)
            elif opt.af_mode == 'gated_state':
                zr_w = tf.get_variable("af_zr_w", [full_size, full_size])
                zr_b = tf.get_variable("af_zr_b", [full_size])
                zr = tf.sigmoid(tf.matmul(outputs, zr_w) + zr_b)
                z = tf.slice(zr, [0, 0], [-1, opt.state_size],
                             name="af_z_gate")
                r = tf.slice(zr, [0, opt.state_size], [-1, -1],
                             name="af_r_gate")
                l = tf.mul(l, r)
                outputs = tf.concat(1, [l, rnn_state])
                h_w = tf.get_variable("af_h_w",
                                      [full_size, opt.state_size])
                h_b = tf.get_variable("af_h_b", [opt.state_size])
                h = tf.tanh(tf.matmul(outputs, h_w) + h_b)
                outputs = tf.mul((1-z), rnn_state) + tf.mul(z, h)
        return outputs, opt.state_size

    def _extract_features(self, opt, l):
        # TODO: placeholder for later refactor
        af_function = 'lm_emb'
        if opt.is_set('af_function'):
            af_function = opt.af_function

        if af_function == 'lm_emb':
            l = tf.nn.embedding_lookup(self._emb_vars, l)
        elif af_function == 'ex_emb':
            l = tf.nn.embedding_lookup(opt.af_ex_emb_vars, l)
        elif af_function == 'emb':
            with tf.variable_scope("afeatures"):
                initializer = tf.uniform_unit_scaling_initializer(
                    dtype=tf.float32)
                # use "lookup" to distinguish between LM embedding
                af_emb_vars = []
                if opt.af_emb_train_vocab_size > 0:
                    af_train_emb = tf.get_variable(
                        "af_train_lookup", [opt.af_emb_train_vocab_size,
                                            opt.af_emb_size],
                        initializer=initializer, dtype=tf.float32)
                    af_emb_vars.append(af_train_emb)
                if opt.af_emb_fix_vocab_size > 0:
                    af_fixed_emb = tf.get_variable(
                        "af_fix_lookup", [opt.af_emb_fix_vocab_size,
                                          opt.af_emb_size],
                        initializer=initializer, dtype=tf.float32, trainable=False)
                    af_emb_vars.append(af_fixed_emb)
                if len(af_emb_vars) > 1:
                    self.af_emb_var = tf.concat(0, af_emb_vars)
                else:
                    self.af_emb_var = af_emb_vars
                l = tf.nn.embedding_lookup(self.af_emb_var, l)
        self._af_size = l.get_shape()[2].value
        if 'emb' in af_function:
                if self.is_training and opt.emb_keep_prob < 1.0:
                    l = tf.nn.dropout(l, opt.emb_keep_prob)
        # TODO: add char-CNN
        return l

    def _get_additional_variables(self):
        return find_trainable_variables(self._top_scope, "afeatures")
