import os
import torch
import time
import cv2
import numpy as np
from PIL import Image
import argparse
from safetensors.torch import load_file
import torch.nn as nn
import sys
import os
from models.unet_conditional import UNet2DConditionModel
from models.controlnet import ControlNetModel
from models.pipeline_controlnext_inpaint import StableDiffusionControlNextInpaintPipeline
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from transformers import AutoTokenizer, PretrainedConfig
import os
import sys
import torch.nn as nn

from ip_adapter import IPAdapter

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




controlnext_path = '/home/chaos/Documents/Chaos_project/model/controlnext/controlnet.safetensors'
unet_path = '/home/chaos/Documents/Chaos_project/model/sd_model/stable_diffusion/unet'
load_weight_increasement = '/home/chaos/Documents/Chaos_project/model/controlnext/unet.safetensors'
image_model_path= '/home/chaos/Documents/Chaos_project/model/sd_model/stable_diffusion/clip_image/'
ip_adapter_weight_path= '/home/chaos/Documents/Chaos_project/model/sd_model/ip-adapter_sd15.bin'
device = 'cuda'

controlnext = ControlNetModel()
load_safetensors(controlnext,controlnext_path)


# pipeline = StableDiffusionControlNextInpaintPipeline()
unet =UNet2DConditionModel.from_pretrained(
        f"{unet_path}", 
        subfolder="unet", 
    )

# load_safetensors(unet,unet_path)
load_safetensors(unet, f'{load_weight_increasement}', strict=False, load_weight_increasement=False)


ip_model = IPAdapter(unet, image_encoder_path=image_model_path,ip_ckpt= ip_adapter_weight_path,device= device, num_tokens=4)
unet = ip_model.unet

class ImageProjWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @property
    def dtype(self):
        # Return the dtype of the model parameters
        return next(self.base_model.parameters()).dtype

# Now wrap your ip_adapter_proj
ip_adapter_proj_wrapped = ImageProjWrapper(ip_model.image_proj_model)

pipe = StableDiffusionControlNextInpaintPipeline.from_single_file('/home/chaos/Documents/Chaos_project/model/sd_model/Realistic_Vision_V6.0_NV_B1.safetensors' , use_safetensors=True, 
                                                                  unet = unet , controlnet = controlnext, 
                                                                  image_encoder = ip_model.image_encoder,
                                                                  ip_adapter_proj = ip_adapter_proj_wrapped
).to('cuda')
# pipe.to('cuda')
pipe.set_progress_bar_config()

prompt = ' I am chaos'
negative_prompt = ' Chaos'
image = Image.open('/home/chaos/Downloads/COCO_train2014_000000122688.jpg')
image.resize((512,512))
pipe(
    prompt=prompt, 
    image=image, 
    original_image=image,
    negative_prompt=negative_prompt, 
    mask_image=image,
    strength=0.99, 
    num_inference_steps=1, 
    callback_steps=1 , # This should be a positive integer, not an image
    ip_adapter_image=image,
)