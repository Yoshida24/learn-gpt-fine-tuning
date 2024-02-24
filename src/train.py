"""_summary_
訓練データと検証データを用いて、
レシピのNER(Named Entity Recognition)モデルをファインチューニングします。
"""

from modules.how_to_finetune_chat_models import train
import argparse


def main():
    # パーサーを作成
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--training_file_id", type=str, help="training_file_id")
    parser.add_argument("--validation_file_id", type=str, help="validation_file_id")

    # 引数を解析
    args = parser.parse_args()

    train.run(
        training_file_id=args.training_file_id,
        validation_file_id=args.validation_file_id,
    )


if __name__ == "__main__":
    main()
