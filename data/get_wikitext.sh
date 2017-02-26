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
  python ../adative_lm/preprocess/preprocess_text.py "$D"
done
