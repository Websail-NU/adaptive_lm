#!/bin/bash

echo "[1/3] Downloading data..."
mkdir ptb_defs
cd ptb_defs
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/ptb_words_defs.tsv
mv ptb_words_defs.tsv train.tsv
mkdir -p gcide/preprocess
mkdir -p wordnet/preprocess
grep "\bgcide\b" train.tsv > gcide/train.tsv
grep "\bwordnet\b" train.tsv > wordnet/train.tsv
echo "[2/3] Preprocessing text files..."
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/preprocess.tar.gz
tar -xf preprocess.tar.gz
rm preprocess.tar.gz
cd ../
awk -F '\t' '{print $1}' ptb_defs/train.tsv | sort | uniq > ptb_defs/preprocess/train_shortlist.txt
python ../preprocess/preprocess_defs.py ptb_defs/ stopwords.txt --bow_vocab_size 2000 --only_train --max_def_len 30
python ../preprocess/preprocess_defs.py ptb_defs/gcide stopwords.txt --bow_vocab_size 2000 --only_train --max_def_len 30
python ../preprocess/preprocess_defs.py ptb_defs/wordnet stopwords.txt --bow_vocab_size 2000 --only_train --max_def_len 30
echo "[3/3] Building definition features..."
cd ../
python -c "import data_utils; data_utils.map_vocab_defs('data/ptb/preprocess/vocab.txt', 'data/ptb_defs/preprocess/', 'data/ptb_defs/preprocess/t_features.pickle')"
python -c "import data_utils; data_utils.map_vocab_defs('data/ptb/preprocess/vocab.txt', 'data/ptb_defs/gcide/preprocess/', 'data/ptb_defs/gcide/preprocess/t_features.pickle')"
python -c "import data_utils; data_utils.map_vocab_defs('data/ptb/preprocess/vocab.txt', 'data/ptb_defs/wordnet/preprocess/', 'data/ptb_defs/wordnet/preprocess/t_features.pickle')"
