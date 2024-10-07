import os
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from diffusers import DDPMScheduler

from pipeline_controlnext_ipadapter_onnx import StableDiffusionControlNextONNXPipeline
from transformers import CLIPTokenizer
import onnxruntime as ort
from configs import *

def log_validation(
    vae, 
    scheduler,
    text_encoder, 
    tokenizer, 
    unet, 
    controlnet, 
    args, 
    device
):
    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    if args.negative_prompt is not None:
        negative_prompts = args.negative_prompt
        assert len(validation_prompts) == len(validation_prompts)
    else:
        negative_prompts = None

    inference_ctx = torch.autocast(device)
    
    pipeline = StableDiffusionControlNextONNXPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        device=device
    )

    image_logs = []
    pil_image = args.pil_image

    if args.pil_image is not None:
        pil_image = Image.open(pil_image).convert("RGB")

    for i, (validation_prompt, validation_image) in enumerate(zip(validation_prompts, validation_images)):
        validation_image = Image.open(validation_image).convert("RGB")
        
        images = []
        negative_prompt = negative_prompts[i] if negative_prompts is not None else None

        for _ in range(args.num_validation_images):
        
            with inference_ctx:
                
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=50, generator=None, negative_prompt=negative_prompt, ip_adapter_image=pil_image, width = args.width, height=args.height,
                )[0]

                images.append(image)

        image_logs.append(
            {"validation_image": validation_image.resize((args.width,args.height)), 
             "ip_adapter_image": pil_image.resize((args.width,args.height)),
             "images": images, "validation_prompt": validation_prompt}
        )

    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    for i, log in enumerate(image_logs):
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        ip_adapter_image = log["ip_adapter_image"]
        validation_image = log["validation_image"]

        formatted_images = []
        formatted_images.append(np.asarray(validation_image))
        formatted_images.append(np.asarray(ip_adapter_image))
    
        for image in images:
            formatted_images.append(np.asarray(image))
            
        for idx, img in enumerate(formatted_images):
            print(f"Image {idx} shape: {img.shape}")
            
        formatted_images = np.concatenate(formatted_images, 1)

        file_path = os.path.join(save_dir_path, "image_{}.png".format(i))
        formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_BGR2RGB)
        print("Save images to:", file_path)
        cv2.imwrite(file_path, formatted_images)

    return image_logs

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pil_image",
        type=str,
        default=None,
        help="IP Adapter image path.",
    )
    
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    return args

if __name__ == "__main__":
    args = parse_args()
    
    device = 'cuda:0'
    
    vae_session = ort.InferenceSession(VAE_ONNX_PATH, providers=providers, provider_options=[{'device_id': 0}])
    
    unet_session = ort.InferenceSession(UNET_ONNX_PATH, providers=providers, provider_options=provider_options)
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
    text_encoder_session = ort.InferenceSession(TEXT_ENCODER_PATH, providers=providers, provider_options=[{'device_id': 0}])
    scheduler = DDPMScheduler.from_pretrained(SCHEDULER_PATH)
        
    controlnet = ort.InferenceSession(CONTROLNEXT_ONNX_PATH, providers=providers, provider_options=provider_options)
    image_encoder = ort.InferenceSession('/home/tiennv/chaos/output/optimize/model.onnx', providers=providers, provider_options=[{'device_id': 0}])
    image_proj_model = ort.InferenceSession(PROJ_ONNX_PATH, providers=providers, provider_options=provider_options)
        
    log_validation(
        vae=vae_session, 
        scheduler=scheduler,
        text_encoder=text_encoder_session, 
        tokenizer=tokenizer, 
        unet=unet_session, 
        controlnet=controlnet, 
        args=args, 
        device=device
    )