import os
import random
import argparse
import json
import itertools
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from model.diffusion.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_ref
from model.diffusion.unet_hacked_garmnet import UNet2DConditionModel

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple,List
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF

from typing import Optional, Union, List, Dict

class TryOffInferencePipeline:
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: Optional[CLIPTextModel] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        image_encoder: Optional[torch.nn.Module] = None,
        image_proj_model: Optional[torch.nn.Module] = None,
        unet_encoder: Optional[UNet2DConditionModel] = None,
        noise_scheduler: Optional[DDPMScheduler] = None,
    ):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.image_encoder = image_encoder
        self.image_proj_model = image_proj_model
        self.unet_encoder = unet_encoder
        self.noise_scheduler = noise_scheduler

    def __call__(
        self,
        image: torch.Tensor,  # [B, 3, H, W]
        im_mask: torch.Tensor,  # [B, 3, H, W]
        inpaint_mask: torch.Tensor,  # [B, 1, H, W]
        cloth_pure: torch.Tensor,  # [B, 3, H, W]
        cloth_trim: torch.Tensor,  # [B, 3, H, W]
        caption: List[str],
        caption_cloth: List[str],
        height: int = 512,
        width: int = 384,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        device = image.device
        dtype = self.vae.dtype

        # --- Encode inputs ---
        ori_person = self.vae.encode(image.to(dtype=dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        masked_person = self.vae.encode(im_mask.to(dtype=dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        mask = F.interpolate(inpaint_mask, size=(height // 8, width // 8))
        mask = mask.reshape(-1, 1, height // 8, width // 8)

        latent_refnet_input = torch.cat([ori_person, mask, masked_person], dim=1)

        model_cloth_input = self.vae.encode(cloth_pure.to(dtype=dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        # --- Add noise ---
        bsz = model_cloth_input.shape[0]
        noise = torch.randn_like(model_cloth_input)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        noisy_latents = self.noise_scheduler.add_noise(model_cloth_input, noise, timesteps)

        # --- Tokenize captions ---
        text_input_ids = self.tokenizer(caption_cloth, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        text_input_ids_2 = self.tokenizer_2(caption_cloth, max_length=self.tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        encoder_output = self.text_encoder(text_input_ids, output_hidden_states=True)
        encoder_output_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True)
        text_embeds = encoder_output.hidden_states[-2]
        pooled_text_embeds = encoder_output_2[0]
        text_embeds_2 = encoder_output_2.hidden_states[-2]
        encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1)

        # --- Time IDs ---
        def compute_time_ids(original_size, crops_coords_top_left=(0, 0)):
            target_size = (height, height)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            return torch.tensor([add_time_ids], device=device)

        add_time_ids = torch.cat([compute_time_ids((height, height)) for _ in range(bsz)])

        # --- Image embeddings ---
        img_emb_list = [cloth_trim[i].unsqueeze(0) for i in range(bsz)]
        image_embeds = torch.cat(img_emb_list, dim=0)
        image_embeds = self.image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
        ip_tokens = self.image_proj_model(image_embeds)

        unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids, "image_embeds": ip_tokens}

        # --- Ref branch caption ---
        text_input_ids = self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        text_input_ids_2 = self.tokenizer_2(caption, max_length=self.tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        encoder_output = self.text_encoder(text_input_ids, output_hidden_states=True)
        encoder_output_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True)
        text_embeds_ref = torch.concat([encoder_output.hidden_states[-2], encoder_output_2.hidden_states[-2]], dim=-1)

        # --- Encode RefNet features ---
        _, reference_features = self.unet_encoder(latent_refnet_input, timesteps, text_embeds_ref, return_dict=False)
        reference_features = list(reference_features)

        # --- Predict noise ---
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs, garment_features=reference_features).sample
        return noise_pred

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        if self.text_encoder_2:
            self.text_encoder_2.save_pretrained(os.path.join(save_directory, "text_encoder_2"))
        self.noise_scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.tokenizer_2.save_pretrained(os.path.join(save_directory, "tokenizer_2"))
        self.image_encoder.save_pretrained(os.path.join(save_directory, "image_encoder"))

    @classmethod
    def from_pretrained(cls, load_directory):
        unet = UNet2DConditionModel.from_pretrained(os.path.join(load_directory, "unet"))
        vae = AutoencoderKL.from_pretrained(os.path.join(load_directory, "vae"))
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(load_directory, "text_encoder"))
        text_encoder_2 = CLIPTextModel.from_pretrained(os.path.join(load_directory, "text_encoder_2"))
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        scheduler = DDPMScheduler.from_pretrained(os.path.join(load_directory, "scheduler"))

        return cls(unet, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, noise_scheduler=scheduler)
