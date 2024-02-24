"""_summary_
このセクションでは、ファインチューニングされたモデルを使用して、
レシピのタイトルと材料から一般的な材料を抽出する推論を行います。
"""

from modules.how_to_finetune_chat_models import inference

import argparse


def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--fine_tuned_model_id", type=str, help="fine-tuned model ID")
    args = parser.parse_args()

    inference.run(
        fine_tuned_model_id=args.fine_tuned_model_id,
        content="Title: Beef Brisket\n"
        "\n"
        'Ingredients: ["4 lb. beef brisket", "1 c. catsup", "1 c. water", '
        '"1/2 onion, minced", "2 Tbsp. cider vinegar", "1 Tbsp. prepared '
        'horseradish", "1 Tbsp. prepared mustard", "1 tsp. salt", "1/2 '
        'tsp. pepper"]\n'
        "\n"
        "Generic ingredients: ",
    )


if __name__ == "__main__":
    main()
