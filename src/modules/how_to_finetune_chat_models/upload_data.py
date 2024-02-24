import json
import pandas as pd
from pandas import DataFrame
from pprint import pprint
from modules.how_to_finetune_chat_models import common


def load_recipes_data() -> DataFrame:
    dir = "src/modules/how_to_finetune_chat_models/data/"
    recipe_df = pd.read_csv(f"{dir}cookbook_recipes_nlg_100.csv")
    pprint(recipe_df.head())
    return recipe_df


def prepare_example_conversation(row):
    system_message = "You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."

    def create_user_message(row):
        return f"""Title: {row['title']}\n\nIngredients: {row['ingredients']}\n\nGeneric ingredients: """

    messages = []
    messages.append({"role": "system", "content": system_message})
    user_message = create_user_message(row)
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": row["NER"]})

    return {"messages": messages}


def write_jsonl(data_list: list, filename: str, save_dir: str) -> None:
    with open(f"{save_dir}{filename}", "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


def run() -> common.UploadedFiles:
    client = common.init_client()
    recipe_df = load_recipes_data()

    # use the first 100 rows of the dataset for training
    training_df = recipe_df.loc[0:10]

    # apply the prepare_example_conversation function to each row of the training_df
    training_data = training_df.apply(prepare_example_conversation, axis=1).tolist()

    for example in training_data[:5]:
        print(example)

    validation_df = recipe_df.loc[10:20]
    validation_data = validation_df.apply(prepare_example_conversation, axis=1).tolist()

    save_dir = "src/modules/how_to_finetune_chat_models/tmp/"
    training_file_name = "tmp_recipe_finetune_training.jsonl"
    write_jsonl(training_data, training_file_name, save_dir)

    validation_file_name = "tmp_recipe_finetune_validation.jsonl"
    write_jsonl(validation_data, validation_file_name, save_dir)

    with open(f"{save_dir}{training_file_name}", "rb") as training_fd:
        training_response = client.files.create(file=training_fd, purpose="fine-tune")

    training_file_id = training_response.id

    with open(f"{save_dir}{validation_file_name}", "rb") as validation_fd:
        validation_response = client.files.create(
            file=validation_fd, purpose="fine-tune"
        )
    validation_file_id = validation_response.id

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)

    return common.UploadedFiles(
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
    )
