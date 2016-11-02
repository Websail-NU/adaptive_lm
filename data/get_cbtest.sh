#!/bin/bash

echo "[1/3] Downloading data..."
wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
echo "[2/3] Extracting files..."
tar --warning=no-unknown-keyword -xf CBTest.tgz
rm CBTest.tgz
mkdir CBTest/preprocess
echo "[3/3] Preprocessing text files..."
python ../preprocess/preprocess_cbtest.py CBTest/ stopwords.txt
