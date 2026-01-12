# scripts/eval_numeracy_blip.py

import os
import re
import json
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering


# Map number words to integers, including "a"/"an" as 1
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a": 1,
    "an": 1,
}


def parse_numeracy_pairs(prompt: str) -> List[Tuple[int, str]]:
    """
    Extract a list of (count, object_word) pairs from a numeracy prompt.

    Works for examples like:
      - "two boys"
      - "four apples were picked from the tree"
      - "four cameras and three horses"
      - "a backpack, three apples, two strawberries, and a bowl ..."
      - "three couches, two keys, two women, four rabbits and one bowl"

    Strategy:
      - convert to lowercase
      - regex: (number_word|digit) + next token as the noun
      - returns list[(count, noun)]
    """
    text = prompt.lower()

    # Build a regex pattern that matches either a digit or a number word
    number_words_pattern = "|".join(sorted(NUMBER_WORDS.keys(), key=len, reverse=True))
    pattern = rf"\b({number_words_pattern}|\d+)\b\s+([a-z]+)"

    pairs: List[Tuple[int, str]] = []
    for m in re.finditer(pattern, text):
        num_str = m.group(1)
        noun = m.group(2)

        if num_str.isdigit():
            count = int(num_str)
        else:
            count = NUMBER_WORDS.get(num_str, None)

        if count is None:
            continue

        # Very light cleaning on noun (e.g., remove trailing punctuation if any)
        noun = re.sub(r"[^a-z]", "", noun)

        if not noun:
            continue

        pairs.append((count, noun))

    return pairs


def extract_int_from_answer(ans: str) -> Optional[int]:
    """
    Try to extract an integer from BLIP's answer string.

    Examples:
      - "three" -> 3
      - "3" -> 3
      - "there are 4 dogs" -> 4
    """
    ans = ans.strip().lower()

    # First look for explicit digits
    m = re.search(r"\b(\d+)\b", ans)
    if m:
        return int(m.group(1))

    # Then look for number words
    for w, v in NUMBER_WORDS.items():
        if w in ans:
            return v

    return None


def load_metadata(images_dir: str) -> List[Dict]:
    path = os.path.join(images_dir, "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.json not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Generative Numeracy using BLIP-VQA."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing generated numeracy images and metadata.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' or 'cpu'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    metadata = load_metadata(args.images_dir)
    print(f"[INFO] Loaded {len(metadata)} entries from metadata.json")

    # Group images by prompt_id
    per_prompt_images: Dict[int, List[Dict]] = {}
    for item in metadata:
        pid = int(item["prompt_id"])
        per_prompt_images.setdefault(pid, []).append(item)

    print(f"[INFO] Number of unique prompts: {len(per_prompt_images)}")

    # Load BLIP-VQA
    model_name = "Salesforce/blip-vqa-base"
    print(f"[INFO] Loading BLIP-VQA model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

    total_pairs = 0
    correct_pairs = 0
    skipped_pairs = 0

    for pid, items in tqdm(per_prompt_images.items(), desc="Evaluating"):
        # Use the first generated image for this prompt
        item = items[0]
        prompt = item["prompt"]

        # Parse (count, object) pairs from the prompt
        pairs = parse_numeracy_pairs(prompt)
        if not pairs:
            # If we cannot parse any pair, skip this prompt
            continue

        img_path = os.path.join(args.images_dir, item["filename"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        for target_count, obj in pairs:
            # Ask BLIP: "How many {obj} are there?"
            question = f"How many {obj} are there?"
            inputs = processor(image, question, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)

            pred_count = extract_int_from_answer(answer)

            if pred_count is None:
                skipped_pairs += 1
                continue

            total_pairs += 1
            if pred_count == target_count:
                correct_pairs += 1

    if total_pairs == 0:
        print("[WARN] No valid (count, object) pairs evaluated.")
        return

    acc = correct_pairs / total_pairs
    print(f"[RESULT] Generative Numeracy accuracy: {acc:.4f}")
    print(
        f"[INFO] total_pairs={total_pairs}, "
        f"correct_pairs={correct_pairs}, "
        f"skipped_pairs={skipped_pairs}"
    )


if __name__ == "__main__":
    main()
