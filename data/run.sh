#!/usr/bin/env bash

# Set up autograder files

python3 load_data.py
python3 tokenize_data.py
python3 create_tests.py