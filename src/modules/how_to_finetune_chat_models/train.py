import time
from modules.how_to_finetune_chat_models import common


def run(training_file_id: str, validation_file_id) -> None:
    client = common.init_client()
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=common.MODEL_NAME,
        suffix=common.FINETUNE_MODEL_SUFFIX,
        hyperparameters={"n_epochs": common.N_EPOCH},
    )

    job_id = response.id
    print("Job ID:", job_id)
    print("Status:", response.status)
    print(f"Job start. See progress on:\nhttps://platform.openai.com/finetune/{job_id}")
