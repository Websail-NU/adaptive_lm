#!/bin/bash

wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
tar -xvf CBTest.tgz
rm CBTest.tgz
mkdir CBTest/preprocess
python ../preprocess/preprocess_cbtest.py CBTest/
