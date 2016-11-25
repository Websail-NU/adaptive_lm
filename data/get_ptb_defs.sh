#!/bin/bash

echo "[1/2] Downloading data..."
mkdir ptb_defs
cd ptb_defs
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/ptb_words_defs.tsv
mv ptb_words_defs.tsv train.tsv
cd ../
echo "[2/2] Preprocessing text files..."
mkdir ptb_defs/preprocess
python ../preprocess/preprocess_defs.py ptb_defs/ stopwords.txt --bow_vocab_size 2000 --only_train --max_def_len 30
