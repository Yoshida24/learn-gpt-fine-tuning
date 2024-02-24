import time
from modules.how_to_finetune_chat_models import common


def run(training_file_id: str, validation_file_id) -> None:
    client = common.init_client()
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=common.MODEL_NAME,
        suffix=common.FINETUNE_MODEL_SUFFIX,
    )

    job_id = response.id

    print("Job ID:", response.id)
    print("Status:", response.status)

    response = client.fine_tuning.jobs.retrieve(job_id)
    print("Job ID:", response.id)
    print("Status:", response.status)
    print("Trained Tokens:", response.trained_tokens)

    for i in range(0, 100):
        response = client.fine_tuning.jobs.list_events(job_id)

        events = response.data
        events.reverse()

        for event in events:
            print(event.message)
            if event.message == "The job has successfully completed":
                break
        time.sleep(5)

    response = client.fine_tuning.jobs.retrieve(job_id)
    fine_tuned_model_id = response.fine_tuned_model

    if fine_tuned_model_id is None:
        raise RuntimeError(
            "Fine-tuned model ID not found. Your job has likely not been completed yet."
        )

    print("Fine-tuned model ID:", fine_tuned_model_id)
