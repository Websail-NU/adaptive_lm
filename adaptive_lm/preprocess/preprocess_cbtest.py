import os
import argparse
import json
import operator

# sos_symbol = '<s>'
eos_symbol = '</s>'
unk_symbol = '<unk>'

parser = argparse.ArgumentParser()
parser.add_argument("cbtest_dir", help="CBTest directory")
parser.add_argument("stopword_file", help="stopword file path")
parser.add_argument("--bow_vocab_size", help="Number of vocab in BOW features",
                    default=5000)
args = parser.parse_args()

addition_test_authors = set((
    'Charles_Kingsley', 'Harriet_Elizabeth_Beecher_Stowe', 'Oscar_Wilde',
    'Washington_Irving'
))
splits = ('train', 'valid', 'test')
ofps = {}
for s in splits:
    ofps[s] = open(os.path.join(args.cbtest_dir,
                                'preprocess/{}.jsonl'.format(s)), 'w')

book = None
w_count = {
    # sos_symbol:0,
    eos_symbol:0,
    unk_symbol:0
}
w_low_count = w_count.copy()
for s in splits:
    f = os.path.join(args.cbtest_dir, 'data/cbt_{}.txt'.format(s))
    with open(f) as ifp:
        for line in ifp:
            if line.startswith('_BOOK_TITLE_'):
                if book is not None:
                    json.dump(book, ofp)
                    ofp.write('\n')
                parts = line.split(':')[1].strip().split('___')
                meta = {'title':parts[1].split('.')[0],
                        'author':parts[0]}
                key = '{}-{}'.format(meta['author'], meta['title'])
                book = {'meta':meta, 'key': key, 'lines':[]}
                ofp = ofps[s]
                if s == 'train' and meta['author'] in addition_test_authors:
                    ofp = ofps['test']
            else:
                line = line.strip()
                book['lines'].append(line)
                # w_count[sos_symbol] += 1
                w_count[eos_symbol] += 1
                # w_low_count[sos_symbol] += 1
                w_low_count[eos_symbol] += 1
                for token in line.split():
                    w_count[token] = w_count.get(token, 0) + 1
                    l_token = token.lower()
                    w_low_count[l_token] = w_low_count.get(l_token, 0) + 1

for ofp in ofps:
    ofps[ofp].close()

w_count = sorted(w_count.items(), key=operator.itemgetter(1), reverse=True)
vocab_filepath = os.path.join(args.cbtest_dir, 'preprocess/vocab.txt')
with open(vocab_filepath, 'w') as ofp:
    for w in w_count:
        ofp.write('{}\t{}\n'.format(w[0], w[1]))

w_low_count = sorted(w_low_count.items(), key=operator.itemgetter(1), reverse=True)
bow_vocab_size = 0
stopwords = set()
with open(args.stopword_file) as ifp:
    for line in ifp:
        stopwords.add(line.strip())

bow_vocab_filepath = os.path.join(args.cbtest_dir, 'preprocess/bow_vocab.txt')
with open(bow_vocab_filepath, 'w') as ofp:
    for w in w_low_count:
        if bow_vocab_size >= args.bow_vocab_size:
            break
        if w[0] in stopwords:
            continue
        ofp.write('{}\t{}\n'.format(w[0], w[1]))
        bow_vocab_size += 1
