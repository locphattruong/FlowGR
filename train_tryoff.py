import os
import random
import argparse
import json
import itertools
from datetime import datetime
import torch
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
from model.tryoff_pipeline import StableDiffusionXLPipeline as TryOffInferencePipeline

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple,List
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF
from model.dataset_train import VitonHDDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--pretrained_garmentnet_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--checkpointing_epoch",type=int,default=100,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--pretrained_ip_adapter_path",type=str,default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",)
    parser.add_argument("--image_encoder_path",type=str,default="ckpt/image_encoder",required=False,help="Path to CLIP image encoder",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--visual_checking",action="store_true",help="Whether or not to perform inference on testset for visually checking during training progress.",)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--logging_steps",type=int,default=1000,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--output_dir",type=str,default="output",help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--snr_gamma",type=float,default=None,help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--num_tokens",type=int,default=16,help=("IP adapter token nums"),)
    parser.add_argument("--learning_rate",type=float,default=1e-5,help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--train_batch_size", type=int, default=6, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=130)
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--seed", type=int, default=42,)    
    parser.add_argument("--num_inference_steps",type=int,default=30,)    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--data_dir", type=str, default="/home/omnious/workspace/yisol/Dataset/VITON-HD/zalando", help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def main():


    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,subfolder="vae",torch_dtype=torch.float16,)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None
    print("UNET ENCODER input channels:", unet_encoder.config.in_channels)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    #customize unet start
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_garmentnet_path, subfolder="unet",low_cpu_mem_usage=False, device_map=None)
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"


    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
 
 
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"],strict=True)

    #ip-adapter
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(accelerator.device, dtype=torch.float32)

    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    image_proj_model.requires_grad_(True)

    unet.encoder_hid_proj = image_proj_model

    # Temporarily dont need new conv, will think about it when using pose for conditioning later
    # conv_new = torch.nn.Conv2d(
    #     in_channels=4+4+1+4,
    #     out_channels=unet.conv_in.out_channels,
    #     kernel_size=3,
    #     padding=1,
    # )
    # torch.nn.init.kaiming_normal_(conv_new.weight)  
    # conv_new.weight.data = conv_new.weight.data * 0.  

    # conv_new.weight.data[:, :9] = unet.conv_in.weight.data  
    # conv_new.bias.data = unet.conv_in.bias.data  

    # unet.conv_in = conv_new  # replace conv layer in unet
    # unet.config['in_channels'] = 13  # update config
    # unet.config.in_channels = 13  # update config
    #customize unet end


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device) 
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)


    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    unet.requires_grad_(True)




    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_encoder.enable_gradient_checkpointing()
    unet.train()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_opt = itertools.chain(unet.parameters())


    optimizer = optimizer_class(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="paired",
        size=(args.height, args.width),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=16,
    )
    test_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="paired",
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    unet,image_proj_model,unet_encoder,image_encoder,optimizer,train_dataloader,test_dataloader = accelerator.prepare(unet, image_proj_model,unet_encoder,image_encoder,optimizer,train_dataloader,test_dataloader)
    initial_global_step = 0

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print("SUCCESS")
    # Train!
    model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    first_epoch = 0
    train_loss=0.0
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(image_proj_model):
                # Inference during training to check the visual result
                if args.visual_checking:
                    if global_step % args.logging_steps == 0:
                        if accelerator.is_main_process:
                            with torch.no_grad():
                                with torch.cuda.amp.autocast():
                                    unwrapped_unet= accelerator.unwrap_model(unet)
                                    newpipe = TryOffInferencePipeline.from_pretrained(
                                        args.pretrained_garmentnet_path,
                                        unet=unwrapped_unet,
                                        vae= vae,
                                        scheduler=noise_scheduler,
                                        tokenizer=tokenizer,
                                        tokenizer_2=tokenizer_2,
                                        text_encoder=text_encoder,
                                        text_encoder_2=text_encoder_2,
                                        image_encoder=image_encoder,
                                        unet_encoder = unet_encoder,
                                        torch_dtype=torch.float16,
                                        add_watermarker=False,
                                        safety_checker=None,
                                    ).to(accelerator.device)
                                    with torch.no_grad():
                                        for n_test, sample in enumerate(test_dataloader):
                                            img_emb_list = []
                                            for i in range(sample['cloth_trim'].shape[0]):
                                                img_emb_list.append(sample['cloth_trim'][i])

                                            prompt = sample["caption_cloth"] # "a frontal view photo of " + cloth_annotation

                                            num_prompts = sample['cloth_trim'].shape[0]                                        
                                            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                                            if not isinstance(prompt, List):
                                                prompt = [prompt] * num_prompts
                                            if not isinstance(negative_prompt, List):
                                                negative_prompt = [negative_prompt] * num_prompts

                                            image_embeds = torch.cat(img_emb_list,dim=0) #cloth_trim image: B, 3, 224, 224 IP Adapter
                                            
                                            with torch.inference_mode():
                                                (
                                                    prompt_embeds,
                                                    negative_prompt_embeds,
                                                    pooled_prompt_embeds,
                                                    negative_pooled_prompt_embeds,
                                                ) = newpipe.encode_prompt(
                                                    prompt,
                                                    num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True,
                                                    negative_prompt=negative_prompt,
                                                )
                                                
                                            
                                                prompt = sample["caption"] # "model is wearing " + cloth_annotation
                                                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                                                if not isinstance(prompt, List):
                                                    prompt = [prompt] * num_prompts
                                                if not isinstance(negative_prompt, List):
                                                    negative_prompt = [negative_prompt] * num_prompts


                                                with torch.inference_mode():
                                                    (
                                                        prompt_embeds_p,
                                                        _,
                                                        _,
                                                        _,
                                                    ) = newpipe.encode_prompt(
                                                        prompt,
                                                        num_images_per_prompt=1,
                                                        do_classifier_free_guidance=False,
                                                        negative_prompt=negative_prompt,
                                                    )
                                                


                                                generator = torch.Generator(newpipe.device).manual_seed(args.seed) if args.seed is not None else None #make the generation deterministic
                                                images = newpipe(
                                                    prompt_embeds=prompt_embeds,
                                                    negative_prompt_embeds=negative_prompt_embeds,
                                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                                    num_inference_steps=args.num_inference_steps,
                                                    generator=generator,
                                                    # pose_img = sample['pose_img'],
                                                    text_embeds_person=prompt_embeds_p,
                                                    mask_image=sample['inpaint_mask'],
                                                    person_image=(sample['image']+1.0)/2.0, 
                                                    height=args.height,
                                                    width=args.width,
                                                    guidance_scale=args.guidance_scale,
                                                    ip_adapter_image = image_embeds,
                                                )[0]
                        
                                            for i in range(len(images)):
                                                images[i].save(os.path.join(
                                                    args.output_dir,
                                                    str(global_step)+"_"+str(sample['c_name'][i].split('.')[0])+"_"+"test.jpg"
                                                    )
                                                )
                                            if n_test > 1:
                                                break
                            del unwrapped_unet
                            del newpipe                
                            torch.cuda.empty_cache()
            # ---1. Prepare latents for Reference Branch (RefNet)---
                ori_person = batch["image"].to(dtype=vae.dtype) 
                ori_person = vae.encode(ori_person).latent_dist.sample()
                ori_person = ori_person * vae.config.scaling_factor
                
                masked_person = vae.encode(
                    batch["im_mask"].reshape(batch["image"].shape).to(dtype=vae.dtype)
                ).latent_dist.sample()
                masked_person = masked_person * vae.config.scaling_factor #
                masks = batch["inpaint_mask"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(masks, size=(args.height // 8, args.width // 8))
                    ]
                )
                mask = mask.reshape(-1, 1, args.height // 8, args.width // 8)

                # pose_map = vae.encode(batch["pose_img"].to(dtype=vae.dtype)).latent_dist.sample()
                # pose_map = pose_map * vae.config.scaling_factor
                latent_refnet_input = torch.cat([ori_person, mask, masked_person], dim=1) # B, 9, 64, 48
                
            # ---2. Prepare latents for Main Branch (MainNet)---
                model_cloth_input = batch["cloth_pure"].to(accelerator.device,dtype=vae.dtype)
                model_cloth_input = vae.encode(model_cloth_input).latent_dist.sample()
                model_cloth_input = model_cloth_input * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_cloth_input)

                bsz = model_cloth_input.shape[0]
                timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_cloth_input.device
                    )
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(model_cloth_input, noise, timesteps)
            # ---3. Get conditions for Main Branch (MainNet): Caption + IP Adapter coarser features---
                # Caption for mainnet (cloth)
                text_input_ids = tokenizer(
                    batch['caption_cloth'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids #token id
                text_input_ids_2 = tokenizer_2(
                    batch['caption_cloth'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids #token id 

                # encode text input for refnet 
                encoder_output = text_encoder(text_input_ids.to(accelerator.device), output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat


                def compute_time_ids(original_size, crops_coords_top_left = (0,0)):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.height, args.height) 
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device)
                    return add_time_ids
                
                add_time_ids = torch.cat(
                    [compute_time_ids((args.height, args.height)) for i in range(bsz)]
                )
                
                # prepare coarser features by IP adapter
                img_emb_list = []
                for i in range(bsz):
                    img_emb_list.append(batch["cloth_trim"][i])
                
                image_embeds = torch.cat(img_emb_list,dim=0)
                image_embeds = image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                ip_tokens =image_proj_model(image_embeds)
                
                # add cond
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                unet_added_cond_kwargs["image_embeds"] = ip_tokens
            # ---4. Prepare conditions for Ref Branch: Caption---
                # Caption for refnet (mask+person)
                text_input_ids = tokenizer(
                    batch['caption'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch['caption'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

            
                encoder_output = text_encoder(text_input_ids.to(accelerator.device), output_hidden_states=True)
                text_embeds_cloth = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                text_embeds_2_cloth = encoder_output_2.hidden_states[-2]
                text_embeds_cloth = torch.concat([text_embeds_cloth, text_embeds_2_cloth], dim=-1) # concat
            # ---5. Get conditions for Main Branch (MainNet): Finer features from Refnet---    
                down,reference_features = unet_encoder(latent_refnet_input, timesteps, text_embeds_cloth,return_dict=False)
                reference_features = list(reference_features)
            # ---6. MainNet forward---    
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states,added_cond_kwargs=unet_added_cond_kwargs,garment_features=reference_features).sample

            # ---7. Calculate loss ---
                # Type of loss & prediction: noise, velocity or original image
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_cloth_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_cloth_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                
                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, 1.0)

                optimizer.step()
                optimizer.zero_grad()
                # Load scheduler, tokenizer and models.
                progress_bar.update(1)
                global_step += 1
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        # ---- Finish training iteration ----
            if global_step % args.checkpointing_epoch == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    unwrapped_unet = accelerator.unwrap_model(
                        unet, keep_fp32_wrapper=True
                    )
                    pipeline = TryOffInferencePipeline.from_pretrained(
                                    args.pretrained_garmentnet_path,
                                    unet=unwrapped_unet,
                                    vae= vae,
                                    scheduler=noise_scheduler,
                                    tokenizer=tokenizer,
                                    tokenizer_2=tokenizer_2,
                                    text_encoder=text_encoder,
                                    text_encoder_2=text_encoder_2,
                                    image_encoder=image_encoder,
                                    unet_encoder = unet_encoder,
                                    torch_dtype=torch.float16,
                                    add_watermarker=False,
                                    safety_checker=None,
                                ).to(accelerator.device)
                    save_path = os.path.join(args.output_dir, model_id, f"checkpoint-{global_step}")
                    pipeline.save_pretrained(save_path)
                    del pipeline

                
if __name__ == "__main__":
    main()    
