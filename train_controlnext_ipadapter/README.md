# ⚙️ **Guide training Stable Diffusion Controlnext IPadapter **
## 📡 ** Documents**



## 📝 **install package requirements **

```text
cd convert_controlnext_to_onnx
```

```text
pip install -r requirements.txt
```

## Download weights % dataset for training 

```text
run code in /convert_controlnext_to_onnx/train_controlnext_ipadapter/get_weight_training.ipynb```

## 🚀 **Run code to download dataset and make train file **
### Download dataset. 

```text
gdown https://drive.google.com/uc?id=1nyGA3BTOF_zqv70plK5-MtUQDVwQ1t86 
```


### Make dataset for training


1. Change script

```python
image_folders = [
    "/Desktop/target_image/",
    "/Desktop/controlnext_cond/",
    "/Desktop/original_mask/",
    "/Desktop/masked_image/",
    "/Desktop/ip_adapter_image/"
]
output_csv = "/data/data.csv"

create_csv_from_folders(image_folders, output_csv)
```
2. Command in CLI

```text
python controlnext_training/make_data.py
```


3. change script training in /convert_controlnext_to_onnx/train_controlnext_ipadapter/controlnext_training/scripts.sh

```script
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 train_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \ #name of model
 --output_dir="/content/save_weight" \  #folder to save weights
 --dataset_name=None \ #pass
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --checkpoints_total_limit 3 \
 --checkpointing_steps 400 \ # save checkpoint after n steps
 --validation_steps 400 \
 --num_train_epochs 4 \
 --train_batch_size=1 \
 --controlnext_scale 0.35 \
 --save_load_weights_increaments \
 --unet_model_name_or_path '/content/weightsdir/unet' \
 --ip_adapter_path '/content/weightsdir/ip-adapter_sd15.bin' \
 --clip_vt_model_name_or_path '/content/weightsdir/clip_model' \
  --mixed_precision 'fp16' \
  --dataset_path '' \ #path to dataset
  --text_encoder '' \ #text encoder name or path
  --tokenizer '' \ # tokenizer name of path
```


4. Training 
```bash 
bash script.sh

```



