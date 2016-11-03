import tensorflow as tf
""" Recurrent langauge models. This code is adapted from:
`Rafal Jozefowicz's lm <https://github.com/rafaljozefowicz/lm>`_
"""

# \[T]/ PRAISE THE SUN!
#  |_|
#  | |

def sharded_variable(name, shape, num_shards,
                     dtype=tf.float32, transposed=False):
    # The final size of the sharded variable may be larger than requested.
    # This should be fine for embeddings.
    shard_size = int((shape[0] + num_shards - 1) / num_shards)
    if transposed:
        initializer = tf.uniform_unit_scaling_initializer(
            dtype=dtype, full_shape=[shape[1], shape[0]])
    else:
        initializer = tf.uniform_unit_scaling_initializer(
            dtype=dtype, full_shape=shape)
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

        losses = []
        tower_grads = []
        # variable scope name must be set before creating LM
        with tf.device(device), tf.variable_scope(tf.get_variable_scope()):
            loss = self._forward(x, y, z)
            losses += [loss]
            # if opt.is_training:
            #     cur_grad = self._backward()

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
        with tf.variable_scope("lstm"):
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
