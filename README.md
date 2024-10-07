# ⚙️ **Guide convert Stable Diffusion Controlnext to onnx**
## 📡 ** Documents**



## 📝 **install package requirements **

```text
cd convert_controlnext_to_onnx
```

```text
pip install -r requirements.txt
```

## 🚀 **Run code convert **
### Convert CLIPVisionModelWithProjection model. 

```text
python -m script.convert_clip_model_vision_to_onnx 
```


### Convert Unet, Controlnext, Proj model 

1. Change script

```python
convert_models(
    controlnext_path='controlnet.safetensors',  # name controlnext model
    load_weight_increasement='unet.safetensors',  # load weight additional unet to connect IP adapter with unet
    image_model_path='clip_model',  # folder save CLIPVisionModelWithProjection model, include pytorch_model.bin and config.json
    ip_adapter_weight_path='ip-adapter_sd15.bin',  # weight IP adapter to convert proj model
    output_path='output',  # folder save file onnx. Important: create a folder unet_optimize into output folder to save optimize unet onnx
    opset=16,
    # fp16=True,
    lora_weight_path='ip-adapter-faceid-plus_sd15_lora.safetensors',
    use_safetensors=True,
    unet_folder_path='SG161222/Realistic_Vision_V5.1_noVAE'  # name of stable diffusion model to convert unet to onnx      
)
```
2. Command in cli

```text
python -m script.convert_controlnext_to_onnx.py
```

### Convert VAE Text Encoder 
```text

optimum-cli export onnx -m SG161222/Realistic_Vision_V5.1_noVAE --task text-encoding your_folder_path
```

## 🚀 **Inferences**

```text
cd convert_controlnext_to_onnx/pipeline
```

1. Change script in configs.py

```text
VAE_CONFIGS = '/vae_decoder/config.json'

VAE_ONNX_PATH = 'vae_decoder/model.onnx'

UNET_ONNX_PATH = "unet_optimize/model.onnx"

TOKENIZER_PATH = 'vae_text_encoder/tokenizer'

TEXT_ENCODER_PATH = 'text_encoder/model.onnx'

SCHEDULER_PATH = 'scheduler'

CONTROLNEXT_ONNX_PATH = 'controlnext/model.onnx'

IMAGE_ENCODER_ONNX_PATH = 'image_encoder/model.onnx'

PROJ_ONNX_PATH = 'proj/model.onnx'

providers = ['CUDAExecutionProvider']  #use cuda

provider_options = [{'device_id': 1}]
```
2. Change script in convert_controlnext_to_onnx/pipeline/script.sh

```script
python run_controlnext.py \
 --output_dir "examples/result" \      #folder to save result image                    
 --validation_image "control/deepfashion_caption/condition-0.png"  \  #image to input controlnext, some samples save in convert_controlnext_to_onnx/image_condition
 --validation_prompt "A  beautiful girl raise both hands " \
 --pil_image "/home/tiennv/chaos/run_cnext/ip_adapter/deepfashion.jpg" \  # face image to input IP adapter
 --negative_prompt "low resolution, two head" \
 --num_validation_images 1 \
 --width 512 \
 --height 512
```


3. Inferences
```bash 
bash script.sh

```



