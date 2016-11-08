import tensorflow as tf
""" Recurrent langauge models. This code is adapted from:
`Rafal Jozefowicz's lm <https://github.com/rafaljozefowicz/lm>`_
"""

# \[T]/ PRAISE THE SUN!
#  |_|
#  | |

def find_trainable_variables(key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             ".*{}.*".format(key))

def sharded_variable(name, shape, num_shards,
                     dtype=tf.float32):
    # The final size of the sharded variable may be larger than requested.
    # This should be fine for embeddings.
    shard_size = int((shape[0] + num_shards - 1) / num_shards)
    initializer = tf.uniform_unit_scaling_initializer(dtype=dtype)
    return [tf.get_variable(name + "_%d" % i,
                            [shard_size, shape[1]],
                            initializer=initializer,
                            dtype=dtype)
            for i in range(num_shards)]

class LM(object):

    def __init__(self, opt):
        self.opt = opt
        self.x = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps])
        self.y = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps])
        self.w = tf.placeholder(tf.int32, [opt.batch_size, opt.num_steps])

        # variable scope name must be set before creating LM
        with tf.variable_scope(tf.get_variable_scope()):
            self.loss = self._forward(self.x, self.y, self.w)
            if opt.is_training:
                self.grads, self.vars = self._backward(loss)

        if opt.is_training:
            self.global_step = tf.get_variable("global_step", [], tf.int32,
                                               initializer=tf.zeros_initializer,
                                               trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(opt.learning_rate)
            self.train_op = optimizer.apply_gradients(
                zip(self.grads, self.vars),
                global_step=self.global_step)
        else:
            self.train_op = tf.no_op()


    def _forward(self, x, y, w):
        opt = self.opt
        w = tf.to_float(w)
        # Input
        emb_vars = sharded_variable(
            "emb", [opt.vocab_size, opt.emb_size], opt.num_shards)
        x = tf.nn.embedding_lookup(emb_vars, x)  # [bs, steps, emb_size]
        if opt.is_training and opt.emb_keep_prob < 1.0:
            x = tf.nn.dropout(x, opt.emb_keep_prob)
        inputs = [tf.squeeze(_x, [1])
                  for _x in tf.split(1, opt.num_steps, x)] # steps * [bs, emb_size]
        # RNN
        with tf.variable_scope("rnn") as vs:
            cell = tf.nn.rnn_cell.BasicLSTMCell(opt.state_size)
            if opt.is_training and opt.keep_prob < 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=opt.keep_prob)
            cell_stack = tf.nn.rnn_cell.MultiRNNCell(
                [cell] * opt.num_layers, state_is_tuple=True)
            self.initial_state = cell_stack.zero_state(
                opt.batch_size, tf.float32)
            outputs, state = tf.nn.rnn(
                cell_stack, inputs, initial_state=self.initial_state)
        outputs = tf.reshape(tf.concat(1, outputs), [-1, opt.state_size])
        # Output
        # XXX: why is softmax_w transposed?
        softmax_w = sharded_variable(
            "softmax_w", [opt.vocab_size, opt.state_size], opt.num_shards)
        softmax_b = tf.get_variable("softmax_b", [opt.vocab_size])
        if opt.num_softmax_sampled == 0:
            full_softmax_w = tf.reshape(
                tf.concat(1, softmax_w), [-1, opt.state_size])
            full_softmax_w = full_softmax_w[:opt.vocab_size, :]
            logits = tf.matmul(
                outputs, full_softmax_w, transpose_b=True) + softmax_b
            targets = tf.reshape(y, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, targets) # longest function name ever!!!
        else:
            targets = tf.reshape(y, [-1, 1])
            loss = tf.nn.sampled_softmax_loss(
                softmax_w, softmax_b, tf.to_float(outputs),
                targets, opt.num_softmax_sampled, opt.vocab_size)
        loss = tf.reduce_mean(loss * tf.reshape(w, [-1]))
        return loss

    def _backward(self, loss):
        opt = self.opt
        loss = loss * opt.num_steps

        emb_vars = find_trainable_variables("emb")
        rnn_vars = find_trainable_variables("rnn")
        softmax_vars = find_trainable_variables("softmax")
        all_vars = emb_vars + rnn_vars + softmax_vars
        grads = tf.gradients(loss, all_vars)
        orig_grads = grads[:]
        emb_grads = grads[:len(emb_vars)]
        grads = grads[len(emb_vars):]
        for i in range(len(emb_grads)):
            assert isinstance(emb_grads[i], tf.IndexedSlices)
            emb_grads[i] = tf.IndexedSlices(
                emb_grads[i].values * opt.batch_size, emb_grads[i].indices,
                emb_grads[i].dense_shape)

        rnn_grads = grads[:len(rnn_vars)]
        softmax_grads = grads[len(rnn_vars):]

        rnn_grads, rnn_norm = tf.clip_by_global_norm(rnn_grads, opt.max_grad_norm)
        clipped_grads = emb_grads + rnn_grads + softmax_grads
        assert len(clipped_grads) == len(orig_grads)
        return clipped_grads, all_vars

class ModelOption(object):
    def __init__(self, **kwds):
        self._default_options()
        self.__dict__.update(kwds)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def _default_options(self):
        self.__dict__.update(
            is_training=True,
            batch_size=8,
            num_steps=10,
            num_shards=8,
            num_layers=1,
            learning_rate=0.8,
            max_grad_norm=10.0,
            emb_keep_prob=0.9,
            keep_prob=0.9,
            vocab_size=10001,
            emb_size=100,
            state_size=100,
            num_softmax_sampled=0,
            run_profiler=False,
        )
