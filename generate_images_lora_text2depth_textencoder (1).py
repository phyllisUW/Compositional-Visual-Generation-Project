import os
import csv
import json
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel


def load_prompts_from_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a [-1, 1] float tensor [C, H, W] as a PNG image.
    """
    tensor = (tensor.clamp(-1, 1) + 1) / 2.0
    tensor = (tensor * 255).round().to(torch.uint8)
    img = Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())
    img.save(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="SagiPolaczek/stable-diffusion-2-1-base",
        help=(
            "Base SD model id (Hugging Face). "
            "Use a mirror since stabilityai/stable-diffusion-2-1-base is gated/404 in your account."
        ),
    )
    parser.add_argument("--prompts_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument(
        "--config_type",
        type=str,
        choices=["cfg1", "cfg2", "cfg3"],
        required=True,
        help="cfg1: baseline depth-LoRA; cfg2: config 2 UNet-LoRA; cfg3: config 3 UNet-LoRA + text encoder LoRA.",
    )
    parser.add_argument(
        "--text_lora_ckpt",
        type=str,
        default=None,
        help="For cfg3: path to text-encoder LoRA adapter.",
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["transformer", "multiscale"],
        required=True,
        help="Which text-to-depth predictor to use.",
    )
    parser.add_argument("--num_images_per_prompt", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_text_encoder(
    model_id: str,
    config_type: str,
    text_lora_ckpt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> CLIPTextModel:
    """
    Load text encoder. For cfg3, apply text encoder LoRA.
    For cfg1/cfg2, use frozen base text encoder.
    """
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    )
    
    # Apply LoRA for cfg3
    if config_type == "cfg3":
        if text_lora_ckpt is None:
            raise ValueError("cfg3 requires --text_lora_ckpt")
        text_encoder = PeftModel.from_pretrained(
            text_encoder, text_lora_ckpt
        )
        print(f"[INFO] Loaded text encoder LoRA from {text_lora_ckpt}")
    
    text_encoder.to(device, dtype=dtype).eval()
    return text_encoder


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32

    # Tokenizer & shared text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer"
    )
    text_encoder = load_text_encoder(
        model_id=args.model_id,
        config_type=args.config_type,
        text_lora_ckpt=args.text_lora_ckpt,
        device=device,
        dtype=dtype,
    )

    # VAE & scheduler
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae"
    ).to(device, dtype=dtype)
    vae.eval()

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # UNet + LoRA (with 5-channel conv_in: 4 latents + 1 depth)
    base_unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet"
    )

    # Adjust conv_in from 4 -> 5 channels to match training setup
    old_conv_in = base_unet.conv_in
    new_in_channels = 5
    new_conv_in = nn.Conv2d(
        new_in_channels,
        old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        padding=old_conv_in.padding,
    )
    with torch.no_grad():
        # Copy original weights for the first 4 channels, zero-init the 5th.
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight
        new_conv_in.weight[:, 4:, :, :] = 0.0
        new_conv_in.bias = old_conv_in.bias

    base_unet.conv_in = new_conv_in
    base_unet.config.in_channels = new_in_channels

    # Load LoRA checkpoint on top of the modified UNet
    unet = PeftModel.from_pretrained(base_unet, args.lora_ckpt)
    unet.to(device, dtype=dtype).eval()

    # Depth predictor (lazy import based on predictor_type)
    if args.predictor_type == "transformer":
        from text2depth_transformer_decoder import model as depth_model
    else:
        from text2depth_multiscale import model as depth_model

    depth_model.to(device, dtype=dtype).eval()

    # Load prompts
    prompts = load_prompts_from_csv(args.prompts_csv)
    metadata = []

    # Shared generator for reproducibility
    g = torch.Generator(device=device).manual_seed(args.seed)

    # Generation loop
    for row in tqdm(prompts, desc="Generating"):
        prompt = row.get("caption", row.get("prompt", ""))
        prompt_id = row.get("prompt_id", None)
        category = row.get("category", None)

        # Encode text: conditional and unconditional
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        uncond_inputs = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            cond_out = text_encoder(**text_inputs)
            uncond_out = text_encoder(**uncond_inputs)

            cond_embeds = (
                cond_out.last_hidden_state
                if hasattr(cond_out, "last_hidden_state")
                else cond_out[0]
            )
            uncond_embeds = (
                uncond_out.last_hidden_state
                if hasattr(uncond_out, "last_hidden_state")
                else uncond_out[0]
            )

        # Depth prediction using conditional embeddings
        # We adapt embedding dimension if the depth model expects a smaller size
        # (e.g., 768 vs SD2.1's 1024).
        with torch.no_grad():
            cond_for_depth = cond_embeds
            if hasattr(depth_model, "text_proj"):
                in_dim = depth_model.text_proj.in_features
                if cond_embeds.shape[-1] != in_dim:
                    cond_for_depth = cond_embeds[..., :in_dim]

            if args.predictor_type == "transformer":
                # Expected: [B, 1, 64, 64]
                depth_pred = depth_model(cond_for_depth)
            else:
                # Multi-scale: use the 64 x 64 output
                d16, d32, d64 = depth_model(cond_for_depth)
                depth_pred = d64

        # Normalize depth to [-1, 1]
        depth_cond = (depth_pred * 2.0 - 1.0).to(device=device, dtype=dtype)

        # Diffusion sampling loop (classifier-free guidance)
        for n in range(args.num_images_per_prompt):
            # Sample initial latent noise
            latents = torch.randn(
                1, 4, 64, 64, device=device, dtype=dtype, generator=g
            )
            noise_scheduler.set_timesteps(args.num_inference_steps)
            latents_t = latents * noise_scheduler.init_noise_sigma

            for t in noise_scheduler.timesteps:
                # Duplicate for unconditional + conditional
                latent_input = torch.cat([latents_t] * 2)
                depth_input = torch.cat([depth_cond] * 2)

                latent_input = noise_scheduler.scale_model_input(
                    latent_input, t
                )
                unet_input = torch.cat([latent_input, depth_input], dim=1)

                with torch.no_grad():
                    noise_pred = unet(
                        unet_input,
                        t,
                        encoder_hidden_states=torch.cat(
                            [uncond_embeds, cond_embeds]
                        ),
                    ).sample

                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (
                    noise_text - noise_uncond
                )

                latents_t = noise_scheduler.step(
                    noise_pred, t, latents_t
                ).prev_sample

            # Decode latents to image space
            with torch.no_grad():
                latents_t = 1.0 / 0.18215 * latents_t
                image = vae.decode(latents_t).sample[0]

            # File naming
            if prompt_id is not None:
                img_name = f"p{prompt_id}_{n}.png"
            else:
                img_name = f"sample_{len(metadata)}.png"

            img_path = os.path.join(args.output_dir, img_name)
            save_image(image, img_path)

            metadata.append(
                {
                    "filename": img_name,
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "category": category,
                    "config_type": args.config_type,
                    "predictor_type": args.predictor_type,
                    "lora_ckpt": args.lora_ckpt,
                }
            )

    # Save metadata
    with open(
        os.path.join(args.output_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
