from repo_controlnext.controlnext_training.models.controlnext import ControlNeXtModel
import torch
from repo_controlnext.controlnext_training.pipeline.pipeline_controlnext import StableDiffusionControlNeXtPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


#TEST CONTROLNEXT MODEL
controlnext = ControlNeXtModel()
# controlnext forward() missing 2 required positional arguments: 'sample' and 'timestep' 
sample = torch.rand(1,3,64,64, dtype= torch.float32)
timestep = torch.rand(1, dtype= torch.float32)

result = controlnext.forward(sample, timestep)
# print((result['output'].shape))
'''
output return is a tensor with shape (1,320,4,4)

'''
controlnet = ControlNetModel.from_pretrained("Pbihao/ControlNeXt")

#TEST PIPELINE CONTROLNEXT
# pipeline = StableDiffusionControlNeXtPipeline




