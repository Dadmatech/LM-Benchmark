#!/bin/bash

git clone git@hf.co:datasets/hezarai/arman-ner
#git clone https://github.com/HaniehP/PersianNER ArmanPersianNER
git clone https://github.com/Text-Mining/Persian-NER.git
git clone https://github.com/majidasgari/ParsNER.git

python3 populate.py
