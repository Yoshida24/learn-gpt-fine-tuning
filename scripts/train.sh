#!/bin/bash
. .venv/bin/activate
set -a && . ./.env && set +a

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --training_file_id) training_file_id="$2"; shift ;;
        --validation_file_id) validation_file_id="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python ./src/train.py --training_file_id $training_file_id --validation_file_id $validation_file_id
