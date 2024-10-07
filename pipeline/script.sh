python run_controlnext.py \
 --output_dir "examples/result" \
 --validation_image "control/deepfashion_caption/condition-0.png"  \
 --validation_prompt "A  beautiful girl raise both hands " \
 --pil_image "/home/tiennv/chaos/run_cnext/ip_adapter/deepfashion.jpg" \
 --negative_prompt "low resolution, two head" \
 --num_validation_images 1 \
 --width 512 \
 --height 512
