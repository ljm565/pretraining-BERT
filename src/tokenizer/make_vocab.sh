#!/bin/sh

## setup
dpath=../../data/wiki-split/raw
tpath=../../data/wiki-split/tokenizer
vocab_size=30000

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 vocab_trainer.py --data $dpath/data.all --size $vocab_size --output $tpath/vocab_$vocab_size