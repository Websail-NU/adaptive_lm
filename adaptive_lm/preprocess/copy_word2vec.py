import sys
import cPickle
import numpy as np
from gensim.models.word2vec import Word2Vec

w2v_path = sys.argv[1]
vocab_path = sys.argv[2]
out_path = sys.argv[3]

print('- Loading word2vec...')
w2v = Word2Vec.load_word2vec_format(w2v_path, binary=True)
words = []
print('- Reading vocab...')
with open(vocab_path) as ifp:
    for line in ifp:
        words.append(line.strip().split('\t')[0])
print('- Vocab size: {}'.format(len(words)))
print('- Copying word2vec...')
vocab2vec = np.random.uniform(low=-1.0, high=1.0,
                              size=(len(words), w2v.vector_size))
vocab2vec = vocab2vec / np.linalg.norm(vocab2vec, ord=2, axis=0)
for i, word in enumerate(words):
    if word in w2v:
        vocab2vec[i] = w2v[word]
print('- Writing output...')
with open(out_path, 'w') as ofp:
    cPickle.dump(obj=vocab2vec, file=ofp)
