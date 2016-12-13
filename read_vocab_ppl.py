# with open('data/ptb/m1/valid_vocab_ppl.txt') as ifp:
#     m1 = []
#     for line in ifp:
#         m1.append(line.strip().split('\t'))
#
# with open('data/ptb/m3/valid_vocab_ppl.txt') as ifp:
#     m2 = []
#     for line in ifp:
#         m2.append(line.strip().split('\t'))

with open('data/ptb_defs/preprocess/train_shortlist.txt') as ifp:
   shortlist = set()
   for line in ifp:
       shortlist.add(line.strip())

with open('data/ptb/preprocess/vocab.txt') as ifp:
    tv = {}
    for line in ifp:
        parts = line.strip().split('\t')
        tv[parts[0]] = int(parts[1])

with open('data/ptb_defs/preprocess/vocab.txt') as ifp:
    dv = {}
    for line in ifp:
        parts = line.strip().split('\t')
        dv[parts[0]] = int(parts[1])


a = []
b = []
c = []
d = []
for k in tv:
    if tv[k] < 20 and k in shortlist:
        a.append(k)
    elif k in shortlist:
        b.append(k)

import random

shortlist_list = list(shortlist)
random.shuffle(shortlist_list)
for k in shortlist_list:
    if len(c) < len(a):
        c.append(k)
    else:
        d.append(k)

with open('data/ptb_defs/preprocess/ptb_rare/learn_shortlist.txt', 'w') as ofp:
    for w in a:
        ofp.write('{}\n'.format(w))

with open('data/ptb_defs/preprocess/ptb_rare/fixed_shortlist.txt', 'w') as ofp:
    for w in b:
        ofp.write('{}\n'.format(w))

with open('data/ptb_defs/preprocess/ptb_rare/shortlist_vocab.txt', 'w') as ofp:
    for w in a:
        ofp.write('{}\t{}\n'.format(w, tv[w]))
    for w in b:
        ofp.write('{}\t{}\n'.format(w, tv[w]))

with open('data/ptb_defs/preprocess/random/learn_shortlist.txt', 'w') as ofp:
    for w in c:
        ofp.write('{}\n'.format(w))

with open('data/ptb_defs/preprocess/random/fixed_shortlist.txt', 'w') as ofp:
    for w in d:
        ofp.write('{}\n'.format(w))

with open('data/ptb_defs/preprocess/random/shortlist_vocab.txt', 'w') as ofp:
    for w in c:
        ofp.write('{}\t{}\n'.format(w, tv[w]))
    for w in d:
        ofp.write('{}\t{}\n'.format(w, tv[w]))


# with open('output.tsv', 'w') as ofp:
#     for i, w in enumerate(v):
#         ofp.write('{}\t{}\t'.format(w[0], w[1]))
#         ofp.write('{}\t{}\t'.format(m1[i][1], m1[i][2]))
#         ofp.write('{}\t'.format(m2[i][2]))
#         ofp.write('{}\n'.format(w[0] in shortlist))
