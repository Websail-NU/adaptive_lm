#!/bin/bash

echo "[1/3] Downloading control words..."
wget  http://websail-fe.cs.northwestern.edu/downloads/dictdef/ptb-100-control-words.txt
echo "[2/3] Copying PTB data..."
DROP_RATES="100 80 60 40 20"
SPLITS="train valid test"
for RATE in $DROP_RATES; do
  mkdir -p ptb-100/drop_$RATE/preprocess
  for SPLIT in $SPLITS; do
    cp ptb/$SPLIT.txt ptb-100/drop_$RATE/
  done
done
mv ptb-100-control-words.txt ptb-100/
echo "[3/3] Preprocessing data..."
for RATE in $DROP_RATES; do
  echo "[3/3], [1/5] $RATE"
  python ../preprocess/drop_lines.py ptb-100/drop_$RATE/train.txt $RATE ptb-100/ptb-100-control-words.txt ptb-100/tmp
  mv ptb-100/tmp ptb-100/drop_$RATE/train.txt
  python ../preprocess/preprocess_text.py ptb-100/drop_$RATE stopwords.txt --bow_vocab_size 2000
done
