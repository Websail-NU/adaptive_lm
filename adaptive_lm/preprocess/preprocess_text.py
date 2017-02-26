import os
import argparse
import json
import operator

# sos_symbol = '<s>'
eos_symbol = '</s>'
unk_symbol = '<unk>'

parser = argparse.ArgumentParser()
parser.add_argument("text_dir",
                    help=("Text directory, the text file should be "
                          "tokenized and sentence separated (lines)."))
parser.add_argument("--stopword_file", help="stopword file path",
                    default='stopwords.txt')
parser.add_argument("--bow_vocab_size", help="Number of vocab in BOW features",
                    default=100)
args = parser.parse_args()

splits = ('train', 'valid', 'test')
ofps = {}
for s in splits:
    ofps[s] = open(os.path.join(args.text_dir,
                                'preprocess/{}.jsonl'.format(s)), 'w')

w_count = {
    # sos_symbol:0,
    eos_symbol:0,
    unk_symbol:0
}
w_low_count = w_count.copy()
for s in splits:
    f = os.path.join(args.text_dir, '{}.txt'.format(s))
    # XXX: Should not hard code dataset name
    dataset = {'meta':{'name':'PTB', 'split':s}, 'key': 'ptb', 'lines':[]}
    with open(f) as ifp:
        for line in ifp:
            line = line.strip()
            dataset['lines'].append(line)
            # w_count[sos_symbol] += 1
            w_count[eos_symbol] += 1
            # w_low_count[sos_symbol] += 1
            w_low_count[eos_symbol] += 1
            for token in line.split():
                w_count[token] = w_count.get(token, 0) + 1
                l_token = token.lower()
                w_low_count[l_token] = w_low_count.get(l_token, 0) + 1
    ofp = ofps[s]
    json.dump(obj=dataset, fp=ofp)

for ofp in ofps:
    ofps[ofp].close()

w_count = sorted(w_count.items(), key=operator.itemgetter(1), reverse=True)
vocab_filepath = os.path.join(args.text_dir, 'preprocess/vocab.txt')
with open(vocab_filepath, 'w') as ofp:
    for w in w_count:
        ofp.write('{}\t{}\n'.format(w[0], w[1]))

w_low_count = sorted(w_low_count.items(), key=operator.itemgetter(1), reverse=True)
bow_vocab_size = 0
stopwords = set()
with open(args.stopword_file) as ifp:
    for line in ifp:
        stopwords.add(line.strip())

bow_vocab_filepath = os.path.join(args.text_dir, 'preprocess/bow_vocab.txt')
with open(bow_vocab_filepath, 'w') as ofp:
    for w in w_low_count:
        if bow_vocab_size >= args.bow_vocab_size:
            break
        if w[0] in stopwords:
            continue
        ofp.write('{}\t{}\n'.format(w[0], w[1]))
        bow_vocab_size += 1
