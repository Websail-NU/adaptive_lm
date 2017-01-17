#!/bin/bash

echo "[1/2] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_defs_ptb_shortlist.txt
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_words_defs.tar.gz
tar -xvf common_words_defs.tar.gz
cp -r common_words_defs/split_v1.2 common_defs_v1.2
rm -r common_words_defs
rm common_words_defs.tar.gz
cd common_defs_v1.2
mv common_defs_ptb_shortlist.txt common_defs_v1.2/shortlist/
mkdir preprocess
SPLITS="train valid test"
for SPLIT in $SPLITS; do
  mv $SPLIT".txt" $SPLIT".tsv"
done
CORPORA="gcide wordnet"
for CORPUS in $CORPORA; do
  mkdir -p $CORPUS"/preprocess"
  mkdir -p $CORPUS"/shortlist"
  for SPLIT in $SPLITS; do
    grep "\b$CORPUS\b" $SPLIT".tsv" > $CORPUS/$SPLIT".tsv"
    awk -F '\t' '{print $1}' $CORPUS/$SPLIT".tsv" | sort | uniq > $CORPUS"/shortlist/shortlist_"$SPLIT".txt"
    cat $CORPUS"/shortlist/shortlist_"$SPLIT".txt" >> $CORPUS"/shortlist/tmp.txt"
  done
  sort $CORPUS"/shortlist/tmp.txt" > $CORPUS"/shortlist/shortlist_all.txt"
done
echo "[2/2] Preprocessing text files..."
cd ..
python ../preprocess/preprocess_defs.py common_defs_v1.2 stopwords.txt --bow_vocab_size 2000 --max_def_len 30
for CORPUS in $CORPORA; do
  python ../preprocess/preprocess_defs.py "common_defs_v1.2/"$CORPUS stopwords.txt --bow_vocab_size 2000 --max_def_len 30
done
# echo "[3/3] Building definition features..."
# cd ../
# python -c "import data_utils; data_utils.map_vocab_defs('data/ptb/preprocess/vocab.txt', 'data/ptb_defs/preprocess/', 'data/ptb_defs/preprocess/t_features.pickle')"
# python -c "import data_utils; data_utils.map_vocab_defs('data/ptb/preprocess/vocab.txt', 'data/ptb_defs/gcide/preprocess/', 'data/ptb_defs/gcide/preprocess/t_features.pickle')"
# python -c "import data_utils; data_utils.map_vocab_defs('data/ptb/preprocess/vocab.txt', 'data/ptb_defs/wordnet/preprocess/', 'data/ptb_defs/wordnet/preprocess/t_features.pickle')"
