#!/bin/env bash

pip install -r /requirements.txt

exec python3 /subgen/subgen.py "$@"
