import argparse
import json
import os
from collections import defaultdict

import csv


def load_category_mapping(csv_path: str):
    """
    Read prompt_id -> category mapping from the prompts CSV.
    This CSV was created by prepare_t2i_from_files.py and filter_t2i_split.py
    and must contain columns: prompt_id, category.
    """
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["prompt_id"])
            cat = row.get("category", "Unknown")
            mapping[pid] = cat
    return mapping


def summarize_by_category(clip_scores_path: str, prompts_csv: str):
    with open(clip_scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_prompt_mean = data["per_prompt_mean"]  # prompt_id -> mean score
    cat_map = load_category_mapping(prompts_csv)

    cat_to_scores = defaultdict(list)

    for pid_str, score in per_prompt_mean.items():
        pid = int(pid_str)
        cat = cat_map.get(pid, "Unknown")
        cat_to_scores[cat].append(score)

    print("Category, NumPrompts, MeanCLIP")
    for cat, scores in cat_to_scores.items():
        if not scores:
            continue
        mean_score = sum(scores) / len(scores)
        print(f"{cat},{len(scores)},{mean_score:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize CLIP scores by high-level T2I category."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing metadata.json and clip_scores.json",
    )
    parser.add_argument(
        "--prompts_csv",
        type=str,
        required=True,
        help="CSV used to generate the images (contains prompt_id, category).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clip_scores_path = os.path.join(args.images_dir, "clip_scores.json")
    if not os.path.exists(clip_scores_path):
        raise FileNotFoundError(f"{clip_scores_path} not found")

    summarize_by_category(clip_scores_path, args.prompts_csv)


if __name__ == "__main__":
    main()
