from modules.how_to_finetune_chat_models import common


def run(fine_tuned_model_id: str, content: str) -> str | None:
    client = common.init_client()
    system_prompt = "You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."
    response = client.chat.completions.create(
        model=fine_tuned_model_id,  # ft:gpt-3.5-turbo-1106:personal:your-suffix:1234AbCd
        messages=[
            {
                "content": system_prompt,
                "role": "system",
            },
            {
                "content": content,
                "role": "user",
            },
        ],
        temperature=0,
        max_tokens=500,
    )

    print("")
    print(f"🖥️ System Prompt:\n{system_prompt}")
    print("")
    print(f"🙋Input:\n{content}")
    print("")
    print(f"🤖Answer:\n{response.choices[0].message.content}")
    print("")

    return response.choices[0].message.content
