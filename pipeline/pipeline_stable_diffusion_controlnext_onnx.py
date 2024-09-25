import numpy as np
from typing import Any, Union, List, Optional, Dict, Any, Callable
import logging
logger = logging
from diffusers.image_processor import  VaeImageProcessor
from .utils import StableDiffusionControlNextLoader, OnnxExecute
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer
from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
import inspect
from tqdm import tqdm
from torchvision import transforms


class StableDiffusionControlNextOnnx:

    def __init__(self):

        self.model_path = None
        self.image_output = None 
        self.clip_image_processor = CLIPImageProcessor()


    def from_folder(self, folder:str,
                     clip_text_model:Any= None, 
                     clip_image_model:Any = None,
                     scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler] = None
                     ):
        self.model_path = StableDiffusionControlNextLoader(folder)
        if self.model_path is None:
            return f'Can not load model with path {folder}'
        
        self.clip_text_model_path = clip_text_model
        self.clip_image_model_path = clip_image_model
        self.scheduler = scheduler
        
    def encoder_controlnext(self, image:torch.Tensor, timesteps: np.array):
        vae_processor =  VaeImageProcessor(do_convert_rgb= True, do_resize= True)
        resize_image = vae_processor.resize(image, height = 64, width= 64)
        resize_image= resize_image.to(dtype= torch.float16)
        inputs = {"sample": resize_image.cpu().numpy(), 'timesteps':timesteps }
        self.controlnext_session =  OnnxExecute(self.model_path.controlnext)
        controlnext_encoder = self.controlnext_session(inputs= inputs, device= 'cpu')

        del self.controlnext_session
        return controlnext_encoder[0] 
    
    def encoder_ipadapter(self, image:Image.Image):
        #extract feature of faceID
        app = FaceAnalysis(name=self.model_path.face_analysis, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(512, 512))
        image = np.array(image)
        faces = app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_embeds = np.array(faceid_embeds).astype(np.float16)

        #get hidden states of images
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.clip_image_model_path)
        clip_image = self.clip_image_processor(image, return_tensors="pt").pixel_values
        clip_embedding = clip_image_encoder(clip_image,output_hidden_states=True).hidden_states[-2]
        clip_embedding =(clip_embedding.detach().numpy()).astype(np.float16)
        inputs = {'image_embedding': faceid_embeds, 'clip_embedding':clip_embedding}
        proj_execute = OnnxExecute(self.model_path.proj)
        out_ipadapter = proj_execute(inputs, device= 'cpu')

        del proj_execute

        return out_ipadapter[0]


    def encoder_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: Optional[int],
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str],
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None):

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

       
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_text_model_path)
        if prompt_embeds is None:
            text_inputs = self.tokenizer(prompt,
                                         padding= "max_length", 
                                         max_length= self.tokenizer.model_max_length,
                                         truncation = True,
                                         return_tensors= 'np'
                                         )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            text_encoder_execute = OnnxExecute(self.model_path.text_encoder)
            inputs_positive = {"input_ids":text_input_ids.astype(np.int32)}
            prompt_embeds = text_encoder_execute(inputs=inputs_positive, device= 'cpu')[0]


        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
                    uncond_tokens: List[str]
                    if negative_prompt is None:
                        uncond_tokens = [""] * batch_size
                    elif type(prompt) is not type(negative_prompt):
                        raise TypeError(
                            f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                            f" {type(prompt)}."
                        )
                    elif isinstance(negative_prompt, str):
                        uncond_tokens = [negative_prompt] * batch_size
                    elif batch_size != len(negative_prompt):
                        raise ValueError(
                            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                            " the batch size of `prompt`."
                        )
                    else:
                        uncond_tokens = negative_prompt

                    max_length = prompt_embeds.shape[1]
                    uncond_input = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="np",
                    )
                    inputs_negative = {'input_ids':uncond_input.input_ids.astype(np.int32) }
                    negative_prompt_embeds = text_encoder_execute(inputs= inputs_negative, device='cpu')[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])
        del text_encoder_execute

        return np.expand_dims(prompt_embeds[0], axis=0)
    




    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image_controlnext: Image.Image= None,
            image_ipadapter: Image.Image= None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[np.random.RandomState] = None,
            latents: Optional[np.ndarray] = None,
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
            callback_steps: int = 1,):
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self.encoder_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt
        )

         # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        
        # if scheduler is None
        if self.scheduler is None:
            self.scheduler = LMSDiscreteScheduler()

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # prepare image for controlnex and IP_adapter
        image_controlnext = image_controlnext.resize((height, width))
        to_tensor = transforms.ToTensor()
        image_controlnext = to_tensor(image_controlnext).unsqueeze(0)


        image_ipadapter = image_ipadapter.resize((height, width))


        #get ip_adapter_embedding
        ip_adpater_embeding = self.encoder_ipadapter(image_ipadapter)
        bs_embed, seq_len, _ = ip_adpater_embeding.shape 
        ip_adpater_embeding = np.repeat(ip_adpater_embeding, num_images_per_prompt, axis=1)

        ip_adpater_embeding = np.reshape(ip_adpater_embeding, (bs_embed * num_images_per_prompt, seq_len, -1))

        prompt_embeds = np.concatenate((prompt_embeds,ip_adpater_embeding), axis= 1)

        progress_bar = tqdm(self.scheduler.timesteps)

        #init Unet execute
        unet_execute = OnnxExecute(self.model_path.unet_optime)
        for i, t in enumerate(progress_bar):
            # latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()
            timesteps = np.array([t], dtype= np.float16)
            #get controlnext embedding
            controlnext_embed = self.encoder_controlnext(image_controlnext, timesteps)
     

            inputs_unet = {'sample': latent_model_input.astype(np.float16), 'timesteps': timesteps, 'encoder_hidden_states':prompt_embeds.astype(np.float16), 'controlnext_hidden_states': controlnext_embed  }
            noise_pred = unet_execute(inputs_unet, device= 'cuda')
            noise_pred = noise_pred[0]

            # perform guidance
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # del unet_execute
        latents = 1 / 0.18215 * latents
        print(latents)
#         vae_decoder_execute = OnnxExecute(self.model_path.vae_decoder)
#         images = [
#     vae_decoder_execute(
#         inputs={'latent_sample': latents[i:i+1]}, device='cuda'
#     )[0]
#     for i in range(latents.shape[0])
# ]

#         image = np.concatenate(images)
#         image = np.clip(image / 2 + 0.5, 0, 1)
#         image = image.transpose((0,2,3,1))
#         image = image * 255
#         image = image.astype(np.uint8)
#         image = image[0] 
#         image = Image.fromarray(image)
#         image.show()










        

        







    
        


###################
pipe = StableDiffusionControlNextOnnx()
pipe.from_folder('/home/chaos/Documents/Chaos_project/model/sd_controlnext_fp16_onnx/',
                  clip_image_model='/home/chaos/Documents/Chaos_project/model/sd_model/stable_diffusion/clip_image/',
                  clip_text_model= 'openai/clip-vit-base-patch32'
)
pose = Image.open('/home/chaos/Downloads/pose.jpg')
face = Image.open('/home/chaos/Downloads/face.jpg')
# image = image.resize((512,512))
# out = pipe.encoder_ipadapter(image)
# print(out.shape)

promt = 'a beautiful girl , good image'
negative_promt =  None
# out = pipe.encoder_prompt(prompt= promt, negative_prompt= negative_promt, num_images_per_prompt= 1, do_classifier_free_guidance= False)
# print(out.shape)
pipe(prompt= promt, image_controlnext= pose, image_ipadapter= face, num_inference_steps= 1, negative_prompt= negative_promt )