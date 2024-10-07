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
Convert VAE, Text Encoder model. 

```text
python -m sscript.convert_clip_model_vision_to_onnx 
```


### Convert Unet, Controlnext, Proj model 

1. Change script

'''text
   convert_models(
        controlnext_path='controlnet.safetensors'   # name controlnext model
        load_weight_increasement='unet.safetensors',  # load weight additional unet to connect IP adapter with unet
        image_model_path='clip_model',  # folder save CLIPVisionModelWithProjection model, include pytorch_model.bin and config.json
        ip_adapter_weight_path='ip-adapter_sd15.bin', # weight IP adapter to convert proj model. 
        output_path='output', #folder save file onnx. Important: create a folder unet_optimize into output folder to save optimize unet onnx
        opset=16,  
        # fp16=True, 
        lora_weight_path='ip-adapter-faceid-plus_sd15_lora.safetensors',
        use_safetensors=True,
        unet_folder_path= 'SG161222/Realistic_Vision_V5.1_noVAE')        # name of stable diffusion model to convert unet to onnx      
'''  
     