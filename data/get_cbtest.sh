#!/bin/bash

echo "[1/4] Downloading data..."
wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
echo "[2/4] Extracting files..."
tar --warning=no-unknown-keyword -xf CBTest.tgz
rm CBTest.tgz
echo "[3/4] Preprocessing text files..."
mkdir CBTest/preprocess
python ../adaptive_lm/preprocess/preprocess_cbtest.py CBTest/
echo "[4/4] Serializing data..."
cd ../
python -c "import data_utils; data_utils.serialize_corpus('data/CBTest/preprocess')"
