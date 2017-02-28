import unittest

from adaptive_lm.utils.common import LazyBunch
from adaptive_lm.utils.data import SenLabelIterator
from adaptive_lm.utils.data import load_datasets

class SenLabelIteratorTest(unittest.TestCase):

    def test_smoke(self):
        opt = LazyBunch(data_dir="data/common_defs_v1.2/wordnet/preprocess/",
                        vocab_file='vocab.txt',
                        train_file='train.jsonl',
                        valid_file='valid.jsonl',
                        test_file='test.jsonl')
        data, vocab = load_datasets(opt, iterator_type=SenLabelIterator,
                                    dataset=['valid'])
        self.assertEqual(len(data), 1)
        self.assertTrue(isinstance(data['valid'], SenLabelIterator))
        b, r = 2, 5
        data['valid'].init_batch(b, r)
        batch = data['valid'].next_batch()
        self.assertTrue(batch.enc_inputs is not None)
        self.assertTrue(batch.new)
        self.assertTrue(data['valid'].is_new_sen())
        self.assertEqual(batch.enc_inputs.shape, (b, r))
        self.assertEqual(batch.total, b * (r - 1))

if __name__ == '__main__':
    unittest.main()
