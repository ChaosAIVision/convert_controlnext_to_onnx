from unet import UNet2DConditionModel
import torch
from safetensors.torch import load_file

# unet = UNet2DConditionModel()
# def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
#     if not load_weight_increasement:
#         if safetensors_path.endswith('.safetensors'):
#             state_dict = load_file(safetensors_path)
#         else:
#             state_dict = torch.load(safetensors_path)
#         model.load_state_dict(state_dict, strict=strict)
#     else:
#         if safetensors_path.endswith('.safetensors'):
#             state_dict = load_file(safetensors_path)
#         else:
#             state_dict = torch.load(safetensors_path)
#         pretrained_state_dict = model.state_dict()
#         for k in state_dict.keys():
#             state_dict[k] = state_dict[k] + pretrained_state_dict[k]
#         model.load_state_dict(state_dict, strict=False)

# load_safetensors(unet, '/home/chaos/Documents/Chaos_project/model/controlnext/unet.safetensors', load_weight_increasement= True)


# sample =  torch.randn(1,4,64,64)
# timestep = torch.tensor([5])
# encoder_hidden_states = torch.randn(1,77,768)
# controlnext_hidden_states = torch.randn(1,1280,8,8)
# scale = torch.tensor([1])
# output_controlnext = {'out':controlnext_hidden_states, 'scale': scale}

# out = unet.forward(sample= sample, 
#                          timestep= timestep, encoder_hidden_states= encoder_hidden_states)
                          
# print(out)

import torch
from safetensors.torch import load_file

# Tải file safetensors
model_file = '/home/chaos/Documents/Chaos_project/model/sd_model/Realistic_Vision_V6.0_NV_B1.safetensors'
model_dict = load_file(model_file)

# Lưu các thành phần riêng biệt
for key, value in model_dict.items():
    print(key)
   
