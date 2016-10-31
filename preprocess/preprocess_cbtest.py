import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("cbtest_dir", help="CBTest directory")
args = parser.parse_args()

addition_test_authors = set((
    'Charles_Kingsley', 'Harriet_Elizabeth_Beecher_Stowe', 'Oscar_Wilde',
    'Washington_Irving'
))
splits = ('train', 'valid', 'test')
ofps = {}
for s in splits:
    ofps[s] = open(os.path.join(args.cbtest_dir,
                                'preprocess/cbt_{}.jsonl'.format(s)), 'w')

book = None
for s in splits:
    f = os.path.join(args.cbtest_dir, 'data/cbt_{}.txt'.format(s))
    with open(f) as ifp:
        for line in ifp:
            if line.startswith('_BOOK_TITLE_'):
                if book is not None:
                    json.dump(book, ofp)
                parts = line.split(':')[1].strip().split('___')
                meta = {'title':parts[1].split('.')[0],
                        'author':parts[0]}
                book = {'meta':meta, 'lines':[]}
                ofp = ofps[s]
                if s == 'train' and meta['author'] in addition_test_authors:
                    ofp = ofps['test']
            else:
                book['lines'].append(line.strip())

for ofp in ofps:
    ofps[ofp].close()
