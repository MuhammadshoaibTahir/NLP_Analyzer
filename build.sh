#!/usr/bin/env bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
#!/bin/bash

echo "Installing requirements..."
echo "Downloading NLTK corpora..."
python download_nltk_corpora.py

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm