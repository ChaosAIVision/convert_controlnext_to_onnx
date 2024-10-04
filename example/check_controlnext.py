import torch
from repo_ipadapter.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus 
# from repo_controlnext.controlnext_training.pipeline.pipeline_controlnext import StableDiffusionControlNeXtPipeline
from safetensors.torch import load_file
from repo_controlnext.controlnext_test.models.controlnet import ControlNetModel
from repo_controlnext.controlnext_test.models.pipeline_controlnext import StableDiffusionControlNextPipeline
#TEST CONTROLNEXT MODEL
from pathlib import Path
# controlnext = ControlNeXtModel()
# controlnext forward() missing 2 required positional arguments: 'sample' and 'timestep' 
sample = torch.rand(1,3,64,64, dtype= torch.float32)
timestep = torch.rand(1, dtype= torch.float32)
# from repo_controlnext.controlnext_test.models.unet import UNet2DConditionModel
from repo_diffusers.src.diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from torch.onnx import export

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
unet = UNet2DConditionModel.from_pretrained('/home/chaos/Documents/Chaos_project/model/sd_model/stable_diffusion/unet')
# load_safetensors(unet, '/home/chaos/Documents/Chaos_project/model/controlnext/unet.safetensors', load_weight_increasement= True)

# print(unet.forward())

pipeline = StableDiffusionControlNextPipeline.from_single_file('/home/chaos/Documents/Chaos_project/model/sd_model/Realistic_Vision_V6.0_NV_B1.safetensors', 
                                                              use_safetensors=True
                                                              ,controlnet=model,
                                                              unet = unet)

# load_safetensors(pipeline.unet, '/home/chaos/Documents/Chaos_project/model/controlnext/unet.safetensors', load_weight_increasement= True)


# Load ip_adapter 
clip_weight =  '/home/chaos/Documents/Chaos_project/model/sd_model/clip_image/'
ip_adapter_weight = '/home/chaos/Documents/Chaos_project/model/sd_model/ip-adapter-faceid-plus_sd15.bin'
# pipeline_ip = IPAdapterFaceIDPlus(pipeline,image_encoder_path = clip_weight, ip_ckpt = ip_adapter_weight, device = 'cuda' )
# proj_model = pipeline_ip.image_proj_model
# for k,v  in proj_model.state_dict().items():
#     print(k)
# dtype = torch.float32
# sample =  torch.randn(1,4,64,64, dtype=dtype).to('cpu')
# timestep = torch.tensor([5],dtype=dtype).to('cpu')
# encoder_hidden_states = torch.randn(1,77,768,dtype=dtype).to('cpu')
# out = (pipeline.unet.forward(sample, timestep, encoder_hidden_states, return_dict = False))
# print(out[0].shape)

# pipeline.save_pretrained(save_directory= '/home/chaos/Documents/Chaos_project/model/unet/')
scale = torch.tensor([1])
controlnext_hidden_states = torch.randn(1,1280,8,8)
encoder_hidden_states = torch.randn(1,77,768)

output_controlnext = {'out':controlnext_hidden_states, 'scale': scale}

sample =  torch.randn(1,4,64,64)
timestep = torch.tensor([5])
encoder_hidden_states = torch.randn(1,77,768)
print(pipeline.unet.forward(sample,timestep,encoder_hidden_states, mid_block_additional_residual= controlnext_hidden_states, return_dict = False)[0].shape)

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset: int = 16,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset
        )
if controlnext_hidden_states is None:
    raise ValueError("controlnext_hidden_states is None")

if sample is None:
    raise ValueError("sample is None")

if timestep is None:
    raise ValueError("timestep is None")

if encoder_hidden_states is None:
    raise ValueError("encoder_hidden_states is None")
from typing import Union, Optional
class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self,unet ):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnext_hidden_states: Optional[torch.Tensor],

    ):
        # output_controlnext = {'out':controlnext_hidden_states, 'scale': scale_controlnext}
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            mid_block_additional_residual=controlnext_hidden_states,
            return_dict=False,
        )[0]
        return noise_pred 
model = UNet2DConditionControlNetModel(pipeline.unet)

onnx_export(
    model,
    model_args=(
        sample,
        timestep,
        encoder_hidden_states,
        controlnext_hidden_states,
    ),
    output_path=Path('/home/chaos/Documents/Chaos_project/output_onnx/model.onnx'),
    ordered_input_names=[
        "sample",
        "timesteps",
        "encoder_hidden_states",
        "controlnext_hidden_states"
    ],
    output_names=["output"],
    dynamic_axes={
        "sample": {0: "batch_size", 2: "height", 3: "width"},
        
    },
    opset=16  # Sử dụng phiên bản opset phù hợp
)

# print(f"Mô hình UNet đã được chuyển đổi sang ONNX và lưu tại {output_path}")
