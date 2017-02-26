import abc
import six
import numpy as np
import tensorflow as tf

@six.add_metaclass(abc.ABCMeta)
class RNNLM(object):
    """A recurrent neural network langauge model abstract interface."""
    @property
    def batch_size(self):
        """The batch size of the inputs."""
        raise NotImplementedError

    @property
    def num_steps(self):
        """The number of unrolling of the RNN cell."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self):
        """Define input variables for the model.

        Args:
            name: Name scope for any created operations.

        Returns:
            `(input_placeholders, initial_state)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self):
        """Define output variables for the model.

        Args:
            name: Name scope for any created operations.

        Returns:
            `(outputs(rnn_output, distributions), final_state)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self):
        """Define loss variables for the model.

        Args:
            name: Name scope for any created operations.

        Returns:
            `(target_placeholders, loss)`.
        """
        raise NotImplementedError

def get_rnn_cell_class(module="tf.contrib.rnn",
                       cell_type=None):
    if cell_type is None:
        cell_type = "BasicLSTMCell"
    return eval("{}.{}".format(module, cell_type))

def get_rnn_cell(state_size, num_stacks,
                    cell_type=None, keep_prob=1.0):
    cell_cls = get_rnn_cell_class(cell_type=cell_type)
    cell = cell_cls(state_size)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=keep_prob)
    cell_stack = tf.contrib.rnn.MultiRNNCell(
        [cell] * num_stacks, state_is_tuple=True)
    return cell_stack
