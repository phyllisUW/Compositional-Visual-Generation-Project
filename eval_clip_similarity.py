import os
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CLIP text-image similarity for generated images."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing generated images and metadata.json")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()

    metadata_path = os.path.join(args.images_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {args.images_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] Loaded {len(metadata)} metadata entries from {metadata_path}")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[INFO] Using device: {device}")

    model_name = "openai/clip-vit-base-patch32"
    print(f"[INFO] Loading CLIP model: {model_name}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    scores = []
    per_prompt_scores = {}

    for item in tqdm(metadata, desc="Evaluating"):
        img_path = os.path.join(args.images_dir, item["filename"])
        prompt = item["prompt"]
        prompt_id = item["prompt_id"]

        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
                dim=-1, keepdim=True
            )
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
                dim=-1, keepdim=True
            )
            sim = (image_embeds @ text_embeds.T).item()

        scores.append(sim)
        per_prompt_scores.setdefault(prompt_id, []).append(sim)

    mean_over_images = sum(scores) / len(scores)
    print(f"[RESULT] Mean CLIP similarity over all images: {mean_over_images:.4f}")

    per_prompt_mean = {
        int(pid): sum(vals) / len(vals)
        for pid, vals in per_prompt_scores.items()
    }
    print(f"[INFO] Computed mean CLIP similarity for {len(per_prompt_mean)} prompts")

    out_path = os.path.join(args.images_dir, "clip_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_over_images": mean_over_images,
                "per_prompt_mean": per_prompt_mean,
            },
            f,
            indent=2
        )
    print(f"[INFO] Saved CLIP scores to {out_path}")


if __name__ == "__main__":
    main()
