from repo_controlnext.controlnext_training.models.controlnext import ControlNeXtModel
import torch
from repo_ipadapter.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus 
# from repo_controlnext.controlnext_training.pipeline.pipeline_controlnext import StableDiffusionControlNeXtPipeline
from safetensors.torch import load_file
from repo_controlnext.controlnext_test.models.controlnet import ControlNetModel
from repo_controlnext.controlnext_test.models.pipeline_controlnext import StableDiffusionControlNextPipeline
#TEST CONTROLNEXT MODEL
from diffusers import StableDiffusionControlNetPipeline
# controlnext = ControlNeXtModel()
# controlnext forward() missing 2 required positional arguments: 'sample' and 'timestep' 
sample = torch.rand(1,3,64,64, dtype= torch.float32)
timestep = torch.rand(1, dtype= torch.float32)

# result = controlnext.forward(sample, timestep)
# print((result['output'].shape))
'''
output return is a tensor with shape (1,320,4,4)

'''

# load model controlnext
model = ControlNetModel()


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

load_safetensors(model, '/home/chaos/Documents/Chaos_project/model/controlnext/controlnet.safetensors')

# for k, v in model.state_dict().items():
#     print(k)


pipeline = StableDiffusionControlNextPipeline.from_single_file('/home/chaos/Documents/Chaos_project/model/sd_model/Realistic_Vision_V6.0_NV_B1.safetensors', 
                                                              use_safetensors=True
                                                              ,controlnet=model)

load_safetensors(pipeline.unet, '/home/chaos/Documents/Chaos_project/model/controlnext/unet.safetensors', load_weight_increasement= True)


# Load ip_adapter 
clip_weight =  '/home/chaos/Documents/Chaos_project/model/sd_model/clip_image/'
ip_adapter_weight = '/home/chaos/Documents/Chaos_project/model/sd_model/ip-adapter-faceid-plus_sd15.bin'
pipeline_ip = IPAdapterFaceIDPlus(pipeline,image_encoder_path = clip_weight, ip_ckpt = ip_adapter_weight, device = 'cuda' )
proj_model = pipeline_ip.image_proj_model
for k,v  in proj_model.state_dict().items():
    print(k)