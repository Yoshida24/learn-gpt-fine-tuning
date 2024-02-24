import openai
from openai import OpenAI
import os
import json
from dataclasses import dataclass

MODEL_NAME = "gpt-3.5-turbo-1106"
FINETUNE_MODEL_SUFFIX = "recipe-ner"
DATASET_DIR = "src/modules/how_to_finetune_chat_models/dataset/"
TRAINDATA_DIR = "src/modules/how_to_finetune_chat_models/train_data/"


@dataclass
class UploadedFiles:
    training_file_id: str
    validation_file_id: str


def init_client() -> OpenAI:
    client = openai.OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"
        )
    )
    return client


def write_jsonl(data_list: list, filename: str, save_dir: str) -> None:
    with open(f"{save_dir}{filename}", "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)
