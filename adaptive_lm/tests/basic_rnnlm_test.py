import unittest
import tensorflow as tf
from adaptive_lm.models.basic_rnnlm import BasicRNNLM
from adaptive_lm.models.basic_rnnlm import DecoderRNNLM

class BasicRNNLMTest(unittest.TestCase):

    def assertStateSize(self, opt, states):
        for state in states:
            for substate in state:
                self.assertEqual(substate.get_shape(), (opt.batch_size, opt.state_size))

    def assertPlaceholderSize(self, opt, var):
        self.assertEqual(var.get_shape(), (opt.batch_size, opt.num_steps))

    def test_smoke_basic_rnn(self):
        tf.reset_default_graph()
        opt = BasicRNNLM.default_model_options()
        m = BasicRNNLM(opt)
        inputs, init_state = m.initialize()
        self.assertPlaceholderSize(opt, inputs.inputs)
        self.assertEqual(inputs.seq_len.get_shape(), (opt.batch_size, ))
        self.assertStateSize(opt, init_state)
        outputs, final_state = m.forward()
        self.assertEqual(len(outputs.rnn_outputs), opt.num_steps)
        self.assertEqual(outputs.distributions.get_shape(), (opt.batch_size, opt.num_steps, opt.vocab_size))
        self.assertStateSize(opt, final_state)
        targets, losses = m.loss()
        self.assertPlaceholderSize(opt, targets.targets)
        self.assertPlaceholderSize(opt, targets.weights)
        self.assertPlaceholderSize(opt, losses.token_loss)
        self.assertEqual(losses.loss.get_shape(), ())

    def test_smoke_decoder_rnn(self):
        tf.reset_default_graph()
        opt = BasicRNNLM.default_model_options()
        m = DecoderRNNLM(opt)
        inputs, init_state = m.initialize()
        self.assertPlaceholderSize(opt, inputs.inputs)
        self.assertPlaceholderSize(opt, inputs.enc_inputs)
        self.assertEqual(inputs.seq_len.get_shape(), (opt.batch_size, ))
        self.assertStateSize(opt, init_state)
        outputs, final_state = m.forward()
        self.assertEqual(len(outputs.rnn_outputs), opt.num_steps)
        self.assertEqual(len(outputs.enc_outputs), opt.num_steps)
        self.assertEqual(outputs.distributions.get_shape(), (opt.batch_size, opt.num_steps, opt.vocab_size))
        self.assertStateSize(opt, final_state)
        targets, losses = m.loss()
        self.assertPlaceholderSize(opt, targets.targets)
        self.assertPlaceholderSize(opt, targets.weights)
        self.assertPlaceholderSize(opt, losses.token_loss)
        self.assertEqual(losses.loss.get_shape(), ())
        # for v in tf.trainable_variables():
        #     print("- {} {} {}".format(v.name, v.get_shape(), v.device))

if __name__ == '__main__':
    unittest.main()
