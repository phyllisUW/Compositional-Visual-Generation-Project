import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model

from tqdm.auto import tqdm

from train_depth_lora_config1 import DepthDataset, parse_args as base_parse_args


def parse_args():
    args = base_parse_args()
    print("=== Using Config 3: LoRA on text-encoder FFN + LoRA U-Net ===")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    set_seed(args.seed)

    # Base SD components
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet"
    )

    # Freeze base weights; only train LoRA adapters
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
        accelerator.print("Enabled xformers memory efficient attention.")
    except Exception as e:
        accelerator.print(f"xformers not enabled: {e}")

    # UNet conv_in -> 5 channels
    accelerator.print(f"Original U-Net conv_in channels: {unet.conv_in.in_channels}")
    old_conv_in = unet.conv_in
    new_in_channels = 5

    new_conv_in = nn.Conv2d(
        new_in_channels,
        old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        padding=old_conv_in.padding,
    )

    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight
        new_conv_in.weight[:, 4:, :, :] = 0.0
        new_conv_in.bias = old_conv_in.bias

    unet.conv_in = new_conv_in
    unet.config.in_channels = 5
    accelerator.print(f"Modified U-Net conv_in channels to: {unet.conv_in.in_channels}")

    # LoRA for UNet
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        modules_to_save=["conv_in"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    # LoRA for text-encoder FFN (fc1/fc2 in CLIP MLP)
    text_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["fc1", "fc2"],
    )
    text_encoder = get_peft_model(text_encoder, text_lora_config)
    text_encoder.print_trainable_parameters()

    # Optimizer: LoRA params of UNet + text encoder
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(text_encoder.parameters()),
        lr=args.learning_rate,
    )

    # Dataset / Dataloader
    dataset = DepthDataset(
        data_root=args.data_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Accelerator
    unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if accelerator.is_main_process:
        accelerator.print("***** Running training (Config 3) *****")
        accelerator.print(f"  Num examples = {len(dataset)}")
        accelerator.print(f"  Num Epochs = {args.num_epochs}")
        accelerator.print(f"  Batch size = {args.train_batch_size}")
        accelerator.print(f"  Grad Accumulation = {args.gradient_accumulation_steps}")
        accelerator.print(f"  Rank (LoRA) = {args.rank}")
        accelerator.print(f"  Steps/epoch = {num_update_steps_per_epoch}")
        accelerator.print(f"  Max train steps = {max_train_steps}")

    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(args.num_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Images -> latents
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, dtype=weight_dtype
                )
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                # Depth maps from dataset
                depth_maps = batch["depth_values"].to(
                    accelerator.device, dtype=weight_dtype
                )

                # Noise schedule
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Concatenate latents and depth along channel dim
                unet_input = torch.cat([noisy_latents, depth_maps], dim=1)

                # Text conditioning (with LoRA adapters)
                input_ids = batch["input_ids"].to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                # UNet forward
                model_pred = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Target = noise or velocity depending on scheduler
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError("Unknown prediction type.")

                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=epoch + 1,
                    loss=float(loss.detach().item()),
                )

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    progress_bar.close()

    if accelerator.is_main_process:
        accelerator.print("Saving Config 3 model...")
        unet.save_pretrained(args.output_dir)
        text_encoder.save_pretrained(args.output_dir + "_text_lora")


if __name__ == "__main__":
    main()

