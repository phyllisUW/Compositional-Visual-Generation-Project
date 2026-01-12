import argparse
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader

# PEFT Imports (Standard LoRA integration)
from peft import LoraConfig, get_peft_model

Image.MAX_IMAGE_PIXELS = None

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Depth-Conditioned LoRA.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="SagiPolaczek/stable-diffusion-2-1-base", # Mirror to avoid 404
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/depth_lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The path to the data directory (root containing index_csvs or the csvs themselves).",
    )
    parser.add_argument("--rank", type=int, default=32, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    
    args = parser.parse_args()
    return args

class DepthDataset(Dataset):
    def __init__(self, data_root, tokenizer, resolution=512):
        # Allow flexibility: if user passes ".../index_csvs", strip it to get the real root
        if data_root.endswith("index_csvs") or data_root.endswith("index_csvs/"):
            self.data_root = os.path.dirname(data_root.rstrip("/"))
        else:
            self.data_root = data_root

        self.tokenizer = tokenizer
        self.resolution = resolution
        
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # CSV Finding Logic
        csv_folder = os.path.join(self.data_root, "index_csvs")
        # Fallback if csvs are in root
        if not os.path.isdir(csv_folder):
            csv_folder = self.data_root

        self.data_frames = []
        if os.path.isdir(csv_folder):
            csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
            csv_files.sort()
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(csv_folder, csv_file))
                    if 'image_path' in df.columns and 'depth_npy' in df.columns:
                        self.data_frames.append(df)
                except Exception as e:
                    pass
        
        if self.data_frames:
            self.data = pd.concat(self.data_frames, ignore_index=True)
            print(f"Dataset: Loaded {len(self.data)} samples from {csv_folder}")
        else:
            raise ValueError(f"No valid CSVs found in {csv_folder}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # ROBUST PATH HANDLING
        # Strip directories from the CSV paths and just use filenames
        img_filename = os.path.basename(str(item['image_path']))
        depth_filename = os.path.basename(str(item['depth_npy']))
        
        # Look in the standard folder structure
        img_path = os.path.join(self.data_root, "images", img_filename)
        depth_path = os.path.join(self.data_root, "depth_64_npy", depth_filename)

        try:
            # Image Loading with Extension Fallback
            if not os.path.exists(img_path):
                if img_path.endswith(".jpg"):
                    alt_path = img_path.replace(".jpg", ".png")
                    if os.path.exists(alt_path): img_path = alt_path
                elif img_path.endswith(".png"):
                    alt_path = img_path.replace(".png", ".jpg")
                    if os.path.exists(alt_path): img_path = alt_path
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transforms(image)
            
            # Tokenize Caption
            caption = str(item['caption_ori']) if 'caption_ori' in item else ""
            inputs = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids = inputs.input_ids[0]

            # Load Depth 
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth not found: {depth_path}")

            depth_map = np.load(depth_path) # (H, W)
            depth_tensor = torch.from_numpy(depth_map).float()
            
            if depth_tensor.ndim == 2:
                depth_tensor = depth_tensor.unsqueeze(0)
            
            # Resize to Latent Size (64x64)
            depth_resized = F.interpolate(
                depth_tensor.unsqueeze(0), size=(64, 64), mode="bicubic", align_corners=False
            ).squeeze(0)
            
            # Normalize to [-1, 1]
            depth_resized = (depth_resized * 2.0) - 1.0

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "depth_values": depth_resized # (1, 64, 64)
            }
        except Exception as e:
            # Recursion breaking: only retry if not at index 0
            if idx != 0:
                return self.__getitem__(0)
            else:
                raise ValueError(f"CRITICAL: Dataset index 0 failed. Check paths! Error: {e}")

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    # OPTIMIZATION: Force Mixed Precision (FP16)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" 
    )
    set_seed(args.seed)

    # Load Models
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")

    # Freeze Base Models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # OPTIMIZATION: Enable Gradient Checkpointing
    unet.enable_gradient_checkpointing()
    
    # OPTIMIZATION: Memory Efficient Attention
    # The G5 (A10G) benefits significantly from xformers
    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention.")
    except:
        pass

    # Modify U-Net Input Channels (4 -> 5)
    print(f"Original U-Net conv_in channels: {unet.conv_in.in_channels}")
    old_conv_in = unet.conv_in
    new_in_channels = 5 # 4 Latents + 1 Depth

    new_conv_in = nn.Conv2d(
        new_in_channels, old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        padding=old_conv_in.padding
    )

    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight
        new_conv_in.weight[:, 4:, :, :] = torch.zeros_like(new_conv_in.weight[:, 4:, :, :])
        new_conv_in.bias = old_conv_in.bias

    unet.conv_in = new_conv_in
    unet.config.in_channels = 5
    print(f"Modified U-Net conv_in channels to: {unet.conv_in.in_channels}")

    # Add LoRA using PEFT
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        modules_to_save=["conv_in"], # Train the 5th channel
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # Data
    dataset = DepthDataset(data_root=args.data_dir, tokenizer=tokenizer, resolution=args.resolution)
    train_dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4
    )

    # Prepare with Accelerator
    # Only unet and optimizer are prepared (wrapped). VAE/TextEncoder are NOT wrapped.
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # OPTIMIZATION: Move Frozen Models to GPU as FP16
    weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if accelerator.is_main_process:
        print(f"***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_epochs}")
        print(f"  Batch size = {args.train_batch_size}")
        print(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Latents (Cast to FP16)
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Depth (Cast to FP16)
                depth_maps = batch["depth_values"].to(dtype=weight_dtype, device=accelerator.device)

                # Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add Noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # CONCATENATE
                unet_input = torch.cat([noisy_latents, depth_maps], dim=1)
                
                # Predict
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                # Loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) 

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
    if accelerator.is_main_process:
        unet.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
