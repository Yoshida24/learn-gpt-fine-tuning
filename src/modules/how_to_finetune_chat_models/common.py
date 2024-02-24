import openai
from openai import OpenAI
import os
import json
from dataclasses import dataclass


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
