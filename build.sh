#!/bin/bash

pip install --upgrade pip
pip install --prefer-binary -r requirements.txt
python download_nltk_corpora.py
python -m spacy download en_core_web_sm