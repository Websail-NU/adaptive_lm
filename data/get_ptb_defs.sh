#!/bin/bash

echo "[1/2] Downloading data..."
mkdir ptb_defs
cd ptb_defs
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/ptb_words_defs.tsv
mv ptb_words_defs.tsv train.tsv
mkdir -p gcide/preprocess
mkdir -p wordnet/preprocess
grep "\bgcide\b" train.tsv > gcide/train.tsv
grep "\bwordnet\b" train.tsv > wordnet/train.tsv
echo "[2/2] Preprocessing text files..."
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/preprocess.tar.gz
tar -xf preprocess.tar.gz
rm preprocess.tar.gz
cd ../
awk -F '\t' '{print $1}' ptb_defs/train.tsv | sort | uniq > ptb_defs/preprocess/train_shortlist.txt
python ../adaptive_lm/preprocess/preprocess_defs.py ptb_defs/ --only_train --max_def_len 30
python ../adaptive_lm/preprocess/preprocess_defs.py ptb_defs/gcide --only_train --max_def_len 30
python ../adaptive_lm/preprocess/preprocess_defs.py ptb_defs/wordnet --only_train --max_def_len 30
