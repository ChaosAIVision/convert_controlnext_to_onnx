
import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import cv2
import json
import time
import pandas as pd
import torch
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModelWithProjection



import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from models.controlnext import ControlNeXtModel
from pipeline.pipeline_controlnext import StableDiffusionControlNeXtPipeline
from models.unet_cond import UNet2DConditionModel

from models.IP_adapter import  ImageProjModel
from safetensors.torch import load_file, save_file
from copy import deepcopy
from ip_adapter.utils import is_torch2_available
from models.IP_adapter import IPAdapter

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")
logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_adapter_modules_unet(unet):
    #check unet is instances DistributedDataParallel  and get original model
    if isinstance(unet, torch.nn.parallel.DistributedDataParallel):
        unet = unet.module  
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules


def init_ip_adapteter(unet_model_name_or_path,ip_adapter_path ):
    proj_model = ImageProjModel()
    unet = UNet2DConditionModel.from_pretrained(
        f"{unet_model_name_or_path}", 
        subfolder="unet", 
    )  
    ip_ckpt = ip_adapter_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adapter_modules = get_adapter_modules_unet(unet)
    ip_apdater = IPAdapter(unet= unet, image_proj_model=proj_model, ckpt_path= ip_ckpt,adapter_modules =adapter_modules)
    return ip_apdater

def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNeXt training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified unet weights are initialized from unet.",
    )
    parser.add_argument(
        '--tokenizer',
        type= str,
        help= 'tokenizer name of path'
    )
    parser.add_argument(
        "--clip_vt_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained clip_vit model or model identifier from huggingface.co/models."
        " If not specified unet weights are initialized from clip_vit.",
    )

    parser.add_argument(
        "--ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip_adapter_path model or model identifier from huggingface.co/models."
        " If not specified unet weights are initialized from ip_adapter_path.",

    )
    parser.add_argument(
        "--load_unet_not_increaments",
        type=str,
        default=None,
        help="Path to pretrained unet module finetune with controlnext",

    )

    

    parser.add_argument(
    "--dataset_path",
    type= str,
    default= None,
    help= 'dataset_path'

    )
    parser.add_argument(
    "--save_embeddings_to_npz",
    type= bool,
    default= False,
    help= 'save_embedding to npz or not'

    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnext-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--controlnext_scale",
        type=float,
        default=1.0,
        help="The control calse for controlnext.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--tensor_dtype_save",
        type=str,
        default='fp16',
        help="The dtype to save data in numpy.",
    )

    parser.add_argument(
        "--path_to_save_data_embedding",
        type=str,
        help="The path to save data embedding.",
    )

    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--save_load_weights_increaments",
        action="store_true",
        help=(
            "whether to store the weights_increaments"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"wandb"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--input_type", type=str,default= 'merge',  help="training with input latent is masked and controlnext is mask else latent is noise and controlnext is masked."
    )
    
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )


    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnext",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnext encoder."
        )

    return args

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    original_values = torch.stack([example['original_values'] for example in examples])
    original_values = original_values.to(memory_format=torch.contiguous_format).float()

    mask_values = torch.stack([example['mask_values'] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()

    ipadapter_images = torch.stack([example['ipadapter_images'] for example in examples])
    ipadapter_images = ipadapter_images.to(memory_format=torch.contiguous_format).float()


    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        'original_values': original_values,
        'mask_values': mask_values,
        "ipadapter_images": ipadapter_images,
    }


def save_embeddings_to_npz(train_dataloader, vae, image_encoder, noise_scheduler, weight_dtype, base_dir, args):
    os.makedirs(base_dir, exist_ok=True)
    if args.tensor_dtype_save == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    npz_data = {} 
    metadata_records = []  
    
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Processing batches"):
        if args.input_type == 'merge':
            latents_target = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents_target = latents_target * vae.config.scaling_factor

            latents = vae.encode(batch["original_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            original_image_mask = batch['mask_values'].to(dtype=weight_dtype)
            original_image_mask_resize = torch.nn.functional.interpolate(
                original_image_mask, size=(latents.shape[2], latents.shape[3])
            )

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device="cpu")

            noisy_full_latent = noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_full_latents_target = noise_scheduler.add_noise(latents_target, noise, timesteps)

            noisy_latents = original_image_mask_resize * noisy_full_latent + (1 - original_image_mask_resize) * latents
            noisy_latent_targets = original_image_mask_resize * noisy_full_latents_target + (1 - original_image_mask_resize) * latents_target

            conditioning_pixel_values = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
        else:
            latents_target = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents_target = latents_target * vae.config.scaling_factor

            masked_latents = vae.encode(batch["original_values"].to(dtype=weight_dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            noise = torch.randn_like(latents_target)
            bsz = latents_target.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device="cpu")

            noisy_latent_targets = noise_scheduler.add_noise(latents_target, noise, timesteps)
            noisy_latents = noise_scheduler.add_noise(masked_latents, noise, timesteps)
            conditioning_pixel_values = batch["original_values"].to(dtype=weight_dtype)

        ip_adapter_condition = batch['ipadapter_images'].to(dtype=weight_dtype)
        clip_embedding = image_encoder(ip_adapter_condition.to('cuda'), return_dict=False)[0]

        npz_data[f"latents_target_{i}"] = latents_target.to(dtype).detach().cpu().numpy()
        npz_data[f"noisy_latents_{i}"] = noisy_latents.to(dtype).detach().cpu().numpy()
        npz_data[f"conditioning_pixel_values_{i}"] = conditioning_pixel_values.to(dtype).detach().cpu().numpy()
        npz_data[f"clip_embedding_{i}"] = clip_embedding.to(dtype).detach().cpu().numpy()
        npz_data[f"timesteps_{i}"] = timesteps.to(dtype).detach().cpu().numpy()
        npz_data[f"noise_{i}"] = noise.to(dtype).detach().cpu().numpy()

        metadata_records.append({
            "batch_index": i,
            "latents_target_key": f"latents_target_{i}",
            "noisy_latents_key": f"noisy_latents_{i}",
            "conditioning_pixel_values_key": f"conditioning_pixel_values_{i}",
            "clip_embedding_key": f"clip_embedding_{i}",
            'timesteps_key': f"timesteps_{i}",
            'noise_key': f"noise_{i}"
        })

    np.savez_compressed(os.path.join(base_dir, "embeddings_data.npz"), **npz_data)
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_parquet(os.path.join(base_dir, "metadata.parquet"))


def load_saved_embeddings(base_dir):
    npz_file = np.load(os.path.join(base_dir, "embeddings_data.npz"))
    metadata_df = pd.read_parquet(os.path.join(base_dir, "metadata.parquet"))

    embedding_data = []
    for _, row in metadata_df.iterrows():
        embedding_data.append({
            "latents_target": npz_file[row["latents_target_key"]],
            "noisy_latents": npz_file[row["noisy_latents_key"]],
            "conditioning_pixel_values": npz_file[row["conditioning_pixel_values_key"]],
            "clip_embedding": npz_file[row["clip_embedding_key"]],
            "timesteps": npz_file[row["timesteps_key"]],
            "noise": npz_file[row["noise_key"]],


        })
    return embedding_data

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
   

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_config=accelerator_project_config,
    )

    accelerator.init_trackers(
        project_name="train_controlnext",
        config={
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "num_train_epochs": args.num_train_epochs,
            "resolution": args.resolution,

        },
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    def find_latest_checkpoint(output_dir):
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]), reverse=True)
        latest_checkpoint = checkpoint_dirs[0] if checkpoint_dirs else None
        return latest_checkpoint
    
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            latest_checkpoint = find_latest_checkpoint(args.output_dir)
            path = latest_checkpoint if latest_checkpoint else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            checkpoint_dir = os.path.join(args.output_dir, path)

            #update controlnext and ip_adapter path
            args.ip_adapter_path = os.path.join(checkpoint_dir, 'ip_adapter.bin')
            args.controlnet_model_name_or_path = os.path.join(checkpoint_dir, 'controlnext.bin')

            optimizer_state_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_state_path):
                optimizer_state = torch.load(optimizer_state_path, map_location=torch.device('cuda'))
                first_epoch = optimizer_state["epoch"]            
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
    else:
        initial_global_step = 0


################
#  LOAD MODEL  #
################
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_model_name_or_path, revision=args.revision, variant=args.variant
    )
    ip_adapter = init_ip_adapteter(args.unet_model_name_or_path, args.ip_adapter_path)
    unet = ip_adapter.unet
    proj_model = ip_adapter.image_proj_model

    #if resume then load some checkpoint unet were trained
    if args.resume_from_checkpoint:
        load_weights_increaments = os.path.join(checkpoint_dir, 'unet.bin')
        load_safetensors(unet,load_weights_increaments, strict= False, load_weight_increasement= False)
    else:
        load_safetensors(unet, args.load_unet_not_increaments, strict= False)
    image_encoder =CLIPVisionModelWithProjection.from_pretrained(args.clip_vt_model_name_or_path)

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnext weights")
        controlnext = ControlNeXtModel()
        load_safetensors(controlnext,args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnext weights from unet")
        controlnext = ControlNeXtModel()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    proj_model.requires_grad_(False)
    controlnext.train()
    unet.train()
    image_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnext).dtype != torch.float32:
        raise ValueError(
            f"Controlnext loaded as datatype {unwrap_model(controlnext).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs


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

    # # Optimizer creation
    params_to_optimize = [
        {
        "params": controlnext.parameters(),
        "lr": args.learning_rate * 1
    },
    # {'params': proj_model.parameters(),
    #  "lr":args.learning_rate * 1}
    ]

    pretrained_trainable_params = {}

    for name, para in unet.named_parameters():
        if "to_out" in name:
            para.requires_grad = True
            para.data = para.to(torch.float32)
            params_to_optimize.append({
                "params": para
            })
        else:
            para.requires_grad = False

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # if args.resume_from_checkpoint:
    #     optimizer.load_state_dict(optimizer_state["optimizer"])


     


################
# INIT DATASET #
################

    from dataset import StableDiffusionDataset
    from dataset_deepfuniture import Deepfurniture_Dataset
    from transformers import  CLIPTokenizer
    # tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer)
    # train_dataset = StableDiffusionDataset(args.dataset_path, input_type ='merge' )
    train_dataset= Deepfurniture_Dataset(args.dataset_path,input_type='not_merge')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer, train_dataloader, lr_scheduler,controlnext,proj_model,unet = accelerator.prepare(
    optimizer, train_dataloader, lr_scheduler,controlnext,proj_model,unet
)
 
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet , image_encoder and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnext = controlnext.to(accelerator.device)
    proj_model = proj_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    def patch_accelerator_for_fp16_training(accelerator):
        org_unscale_grads = accelerator.scaler._unscale_grads_

        def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
            return org_unscale_grads(optimizer, inv_scale, found_inf, True)

        accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

    if args.mixed_precision == "fp16":
        patch_accelerator_for_fp16_training(accelerator)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Input_type = {args.input_type}")


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    image_logs = None
    if args.save_embeddings_to_npz == True:
        save_embeddings_to_npz(train_dataloader, vae,image_encoder,noise_scheduler, weight_dtype,args.path_to_save_data_embedding, args)
    del vae
    del image_encoder
    base_dir = args.path_to_save_data_embedding
    embedding_data = load_saved_embeddings(base_dir)
    for epoch in range(first_epoch, args.num_train_epochs):     
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnext, unet):             
                batch_data = embedding_data[step]
                device= 'cuda'            
                latents_target = torch.tensor(batch_data["latents_target"]).to(dtype=weight_dtype, device =device)
                noisy_latents = torch.tensor(batch_data["noisy_latents"]).to(dtype=weight_dtype, device =device)
                noise = torch.tensor(batch_data["noise"]).to(dtype=weight_dtype, device =device)
                timesteps = torch.tensor(batch_data["timesteps"]).to(dtype=weight_dtype, device =device)
                timesteps = timesteps.long()
                conditioning_pixel_values = torch.tensor(batch_data["conditioning_pixel_values"]).to(dtype=weight_dtype,  device =device)
                if conditioning_pixel_values.shape[1] == 1:
                    conditioning_pixel_values = conditioning_pixel_values.repeat(1, 3, 1, 1)
                clip_embedding = torch.tensor(batch_data["clip_embedding"]).to(dtype=weight_dtype, device =device)
                encoder_hidden_states = proj_model(clip_embedding)
                controlnext_output = controlnext(conditioning_pixel_values, timesteps)
                
               # Sample a random timestep for each image
                          
                # Predict the noise residual
                unet.to(device)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    mid_block_additional_residual=controlnext_output['output'],
                    mid_block_additional_residual_scale= controlnext_output['scale'],
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents_target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                # import torch.nn.utils as nn_utils

                # clip_value = 1.0 

                # for name, para in unet.named_parameters():
                #     if para.requires_grad and para.grad is not None:
                #         nn_utils.clip_grad_norm_(para, clip_value)

            
                params_to_clip = []
                for param_group in params_to_optimize:
                    params_to_clip += param_group["params"]
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
         
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        

                        #save optimizer state
                        optimizer_state_path = os.path.join(save_path, "optimizer.pt")
                        config ={
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch
                        }
                        # torch.save(config, optimizer_state_path)
                        accelerator.save(config, optimizer_state_path)


                        #save controlnext
                        controlnext_path = os.path.join(save_path, 'controlnext.bin')
                        save_controlnext = accelerator.unwrap_model(deepcopy(controlnext))
                        torch.save(save_controlnext.cpu().state_dict(), controlnext_path)
                        del save_controlnext


                        #save unet
                        if not args.save_load_weights_increaments:
                            save_unet = {}
                            unet_state_dict = accelerator.unwrap_model(unet).state_dict()
                            unet_path = os.path.join(save_path, 'unet.bin')
                            for name, paras in pretrained_trainable_params.items():
                                trained_paras = deepcopy(unet_state_dict[name]).detach().cpu()
                                save_unet[name] = trained_paras
                            torch.save(save_unet, unet_path)
                            del save_unet
                            del unet_state_dict

                        if args.save_load_weights_increaments:
                            save_unet = {}
                            unet_state_dict = accelerator.unwrap_model(unet).state_dict()
                            unet_path = os.path.join(save_path, 'unet_weight_increasements.bin')
                            for name, paras in pretrained_trainable_params.items():
                                trained_paras = deepcopy(unet_state_dict[name]).detach().cpu()
                                save_unet[name] = trained_paras - paras
                            torch.save(save_unet, unet_path)
                            del save_unet
                            del unet_state_dict
                            
                        #save ip adapter 
                        # ip_adapter_path = os.path.join(save_path, 'ip_adapter.bin')
                        # save_proj_model = accelerator.unwrap_model(deepcopy(proj_model))
                        # modulist = get_adapter_modules_unet(unet)
                        # combine_weight = {
                        #     'image_proj': save_proj_model.cpu().state_dict(),  
                        #     'ip_adapter': modulist.state_dict()

                        # }

                        # torch.save(combine_weight, ip_adapter_path)
                        # del save_proj_model

                        
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

if __name__ == "__main__":
    args = parse_args()
    main(args)