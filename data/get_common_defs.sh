#!/bin/bash

echo "[1/2] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_defs_ptb_shortlist.txt
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_words_defs.tar.gz
tar -xvf common_words_defs.tar.gz
cp -r common_words_defs/split_v1.2 common_defs_v1.2
rm -r common_words_defs
rm common_words_defs.tar.gz
mv common_defs_ptb_shortlist.txt common_defs_v1.2/shortlist/
cd common_defs_v1.2
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
cd ../
python ../adaptive_lm/preprocess/preprocess_defs.py common_defs_v1.2 --max_def_len 30
for CORPUS in $CORPORA; do
  python ../adaptive_lm/preprocess/preprocess_defs.py "common_defs_v1.2/"$CORPUS --max_def_len 30
done
