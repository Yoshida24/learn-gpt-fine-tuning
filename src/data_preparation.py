"""_summary_
このコードは、レシピデータの前処理+ モデルのファインチューニング用データの準備を行います。
- レシピデータからトレーニングデータとバリデーションデータを生成しJSONL形式でローカルに保存
- OpenAIへトレーニングデータとバリデーションデータをアップロード
"""

from modules.how_to_finetune_chat_models import upload_data


def main():
    upload_data.run()


if __name__ == "__main__":
    main()
