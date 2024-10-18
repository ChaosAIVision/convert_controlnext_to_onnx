import csv
import os

def create_csv(image_paths, text_prompt, output_csv):
    """
    Hàm tạo ra file CSV từ 5 tấm ảnh và 1 text prompt.
    
    Parameters:
    - image_paths (list): Danh sách chứa đường dẫn đến 5 tấm ảnh.
    - text_prompt (str): Text prompt cho các ảnh.
    - output_csv (str): Đường dẫn đến file CSV đầu ra.
    """
    # Kiểm tra danh sách ảnh có đủ 5 phần tử không
    if len(image_paths) != 5:
        raise ValueError("Cần phải có đúng 5 tấm ảnh trong danh sách image_paths.")
    
    # Kiểm tra các đường dẫn ảnh có tồn tại không
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Đường dẫn ảnh {path} không tồn tại.")
    
    # Tạo file CSV và ghi dữ liệu
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Ghi tiêu đề cho các cột
        writer.writerow(["pixel_values", "conditioning_pixel_values", "original_values", "mask_values", "ipadapter_images", "text_prompt"])
        
        # Ghi nội dung vào hàng (5 đường dẫn ảnh + 1 text prompt)
        writer.writerow(image_paths + [text_prompt])
    
    print(f"File CSV đã được tạo thành công: {output_csv}")

# Ví dụ sử dụng hàm
image_paths = [
    "/content/drive/MyDrive/D2T/train_controlnext/image_cond/COCO-train2014-000000122688.jpg",
    "/content/drive/MyDrive/D2T/train_controlnext/image_cond/cnet.jpg",
    "/content/drive/MyDrive/D2T/train_controlnext/image_cond/COCO-train2014-000000122688.jpg",
    "/content/drive/MyDrive/MyDrive/D2T/train_controlnext/image_cond/inpaint-range.jpg",
    "/content/drive/MyDrive/D2T/train_controlnext/image_cond/another_clock.jpg"
]

text_prompt = "A beautiful clock"
output_csv = "/content/dataset/data.csv"

create_csv(image_paths, text_prompt, output_csv)

# from diffusers import CLIPVisionModelWithProjection



#############################################################################
# from models.IP_adapter
# from models.IP_adapter import IPAdapter, ImageProjModel
# from ip_adapter.utils import is_torch2_available

# from diffusers import  UNet2DConditionModel

# if is_torch2_available():
#     from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
# else:
#     from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
# import torch 

# # from models.unet import UNet2DConditionModel
# from models.unet_cond import UNet2DConditionModel

# def get_adapter_modules_unet(unet):
#     attn_procs = {}
#     unet_sd = unet.state_dict()
#     for name in unet.attn_processors.keys():
#         cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#         if name.startswith("mid_block"):
#             hidden_size = unet.config.block_out_channels[-1]
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = unet.config.block_out_channels[block_id]
#         if cross_attention_dim is None:
#             attn_procs[name] = AttnProcessor()
#         else:
#             layer_name = name.split(".processor")[0]
#             weights = {
#                 "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
#                 "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
#             }
#             attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
#             attn_procs[name].load_state_dict(weights)
#     unet.set_attn_processor(attn_procs)
#     adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
#     return adapter_modules


# def init_ip_adapteter(unet_model_name_or_path,ip_adapter_path ):
#     proj_model = ImageProjModel()
#     unet = UNet2DConditionModel.from_pretrained(
#         f"{unet_model_name_or_path}", 
#         subfolder="unet", 
#     )
    
#     ip_ckpt = ip_adapter_path
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     adapter_modules = get_adapter_modules_unet(unet)

#     ip_apdater = IPAdapter(unet= unet, image_proj_model=proj_model, ckpt_path= ip_ckpt,adapter_modules =adapter_modules)
#     return ip_apdater


# ip_adapter = init_ip_adapteter('/home/tiennv/chaos/weight_folder/unet','/home/tiennv/chaos/weight_folder/ip-adapter_sd15.bin' )


# def get_adapter_modules_unet(unet):
#     attn_procs = {}
#     unet_sd = unet.state_dict()
#     for name in unet.attn_processors.keys():
#         cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#         if name.startswith("mid_block"):
#             hidden_size = unet.config.block_out_channels[-1]
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = unet.config.block_out_channels[block_id]
#         if cross_attention_dim is None:
#             attn_procs[name] = AttnProcessor()
#         else:
#             layer_name = name.split(".processor")[0]
#             weights = {
#                 "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
#                 "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
#             }
#             attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
#             attn_procs[name].load_state_dict(weights)
#     unet.set_attn_processor(attn_procs)
#     adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
#     return adapter_modules


# def init_ip_adapteter(unet_model_name_or_path,ip_adapter_path ):
#     proj_model = ImageProjModel()
#     unet = UNet2DConditionModel.from_pretrained(
#         f"{unet_model_name_or_path}", 
#         subfolder="unet", 
#     )
    
#     ip_ckpt = ip_adapter_path
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     adapter_modules = get_adapter_modules_unet(unet)

#     ip_apdater = IPAdapter(unet= unet, image_proj_model=proj_model, ckpt_path= ip_ckpt,adapter_modules =adapter_modules)
#     return ip_apdater
# ip_adapter = init_ip_adapteter('/home/tiennv/chaos/weight_folder/unet','/home/tiennv/chaos/weight_folder/ip-adapter_sd15.bin' )
