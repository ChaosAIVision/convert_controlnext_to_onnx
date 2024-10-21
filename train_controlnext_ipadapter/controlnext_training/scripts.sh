CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 1234 train_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path '/home/tiennv/chaos/weight_folder/controlnet.safetensors' \
 --output_dir="/home/tiennv/chaos/weight_training" \
 --dataset_name=None \
 --resolution=512 \
 --learning_rate=1e-7 \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --checkpoints_total_limit 3 \
 --checkpointing_steps 400 \
 --validation_steps 400 \
 --num_train_epochs 4 \
 --train_batch_size=1 \
 --controlnext_scale 0.35 \
 --save_load_weights_increaments \
 --unet_model_name_or_path '/home/tiennv/chaos/weight_folder/unet' \
 --ip_adapter_path '/home/tiennv/chaos/weight_folder/ip-adapter_sd15.bin' \
 --clip_vt_model_name_or_path '/home/tiennv/chaos/weight_folder/clip_model' \
  --dataset_path '/home/tiennv/chaos/dataset/data.csv' \
  --text_encoder '/home/tiennv/chaos/weight_folder/sd_inpaint/text_encoder' \
  --tokenizer '/home/tiennv/chaos/weight_folder/sd_inpaint/tokenizer' \




# CUDA_VISIBLE_DEVICES=4 python run_controlnext.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --output_dir="test" \
#  --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
#  --controlnet_model_name_or_path checkpoints/checkpoint-1400/controlnext.bin \
#  --unet_model_name_or_path checkpoints/checkpoint-1200/unet.bin \
#  --controlnext_scale 0.35 



# CUDA_VISIBLE_DEVICES=5 python run_controlnext.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --output_dir="test" \
#  --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
#  --controlnet_model_name_or_path checkpoints/checkpoint-400/controlnext.bin \
#  --unet_model_name_or_path checkpoints/checkpoint-400/unet_weight_increasements.bin \
#  --controlnext_scale 0.35 \
#  --save_load_weights_increaments 