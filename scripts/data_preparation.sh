#!/bin/bash
. .venv/bin/activate
set -a && . ./.env && set +a

python ./src/data_preparation.py
