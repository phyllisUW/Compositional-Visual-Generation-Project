# generate_poster_images.py
import os
import csv
import tempfile

# Create temporary CSV with 4 prompts
prompts = [
    "A black and white cat sits in a white sink",
    "A bathroom with red tile and a green shower curtain", 
    "A person is holding a glass of wine and enjoying a sunset",
    "Five drums"
]

temp_csv = "poster_prompts.csv"
with open(temp_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['prompt_id', 'prompt', 'category'])
    writer.writeheader()
    for i, prompt in enumerate(prompts):
        writer.writerow({'prompt_id': i, 'prompt': prompt, 'category': 'poster'})

print(f"Created {temp_csv} with {len(prompts)} prompts")