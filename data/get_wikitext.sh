#!/bin/bash

echo "[1/2] Downloading data..."
DATASETS="wikitext-2 wikitext-103"
SPLITS="train valid test"
VERSION="v1"
for D in $DATASETS; do
  wget "https://s3.amazonaws.com/research.metamind.io/wikitext/$D-$VERSION.zip"
  unzip "$D-$VERSION.zip"
  rm "$D-$VERSION.zip"
  for S in $SPLITS; do
    mv "$D/wiki.$S.tokens" "$D/$S.txt"
  done
  mkdir -p "$D/preprocess"
done
echo "[2/2] Preprocessing..."
for D in $DATASETS; do
  python ../preprocess/preprocess_text.py "$D" stopwords.txt --bow_vocab_size 2000
done
# mkdir ptb/preprocess
# python ../preprocess/preprocess_text.py ptb/ stopwords.txt --bow_vocab_size 2000
# #cd ../
# #python -c "import data_utils; data_utils.serialize_corpus('data/ptb/preprocess')"
