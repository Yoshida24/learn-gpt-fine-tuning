# learn-gpt-fine-tuning

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=762531272&skip_quickstart=true)

Preset for development on Python using venv.

**included:**
- Lint and Format
- Task runner
- Env support

## Usage

depends on:
- Python: 3.11.7
- pip: 24.0
- GNU Make: 3.81

support:
- OS: M1 Macbook Air Ventura 13.4.1


## Gettig Started
First of all, install VSCode recommended extensions. This includes Linter, Formatter, and so on. Recommendation settings is written on `.vscode/extensions.json`.

Duplicate `.env.sample` and rename to `.env` , and input your key and token.

Then, install dependencies:

```bash
source .venv/bin/activate
pip install -r requiments.txt
set -a && source ./.env && set +a
```

Now you can try Fine-tuning.

## Fine-tune by sample dataset
> ref. https://cookbook.openai.com/examples/how_to_finetune_chat_models

Transform dataset to training data and upload training data and validationdata to OpenAI:

```bash
make data_preparation
```

Fine-tune by training data and validation data.

> Note
> `training_file_id` and `validation_file_id` are found on [OpenAI > Storage](https://platform.openai.com/storage).

```bash
make train ARGS="--training_file_id file-xxxxxxxxxxxxxxxxxxxxxxxx --validation_file_id file-yyyyyyyyyyyyyyyyyyyyyyyy"
```

To check fine-tuneed model, inference by your trained model ID:

> Note
> `fine_tuned_model_id` are found on [OpenAI > Fine-tuning](https://platform.openai.com/finetune).

```bash
make inference ARGS="--fine_tuned_model_id ft:gpt-3.5-turbo-1106:personal:your-prefix:zzzzzzzz"
```

You will see answer based on your dataset!

## Develop App
On usual develop, first you activate `venv` first like below.

```bash
source .venv/bin/activate
```

Save requirements:

```bash
pip freeze > requirements.txt
```

Deactivate venv:

```bash
deactivate
```
