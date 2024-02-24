#!/bin/bash
. .venv/bin/activate
set -a && . ./.env && set +a

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fine_tuned_model_id) fine_tuned_model_id="$2"; shift 2 ;;
        *) echo "不明なパラメータが渡されました: $1"; exit 1 ;;
    esac
done

python ./src/inference.py --fine_tuned_model_id $fine_tuned_model_id
