import argparse
import os
import shutil
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from packaging import version
from polygraphy.backend.onnx.loader import fold_constants
from torch.onnx import export

from typing import Union, Optional, Tuple
from diffusers import AutoPipelineForText2Image
from repo_ipadapter.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus, IPAdapterFaceID
from repo_ipadapter.ip_adapter.ip_adapter import IPAdapter
from repo_controlnext.controlnext_test.models.controlnet import ControlNetModel
from repo_controlnext.controlnext_test.models.pipeline_controlnext import StableDiffusionControlNextPipeline
from safetensors.torch import load_file
# from repo_diffusers.src.diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from condition_unet.unet import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection




is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
is_torch_2_0_1 = version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.1")

class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 4147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


def optimize(onnx_graph, name, verbose):
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    opt.fold_constants()
    opt.info(name + ": fold constants")
    # opt.infer_shapes()
    # opt.info(name + ': shape inference')
    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info(name + ": finished")
    return onnx_opt_graph



class ImageProjModel(torch.nn.Module):
    def __init__(self, proj_model):
        super().__init__()
        self.proj_model = proj_model

    def forward(self, clip_embedding: torch.tensor) -> torch.tensor:
        output_proj = self.proj_model(clip_embedding)
        return output_proj

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
        scale_controlnext:Union[torch.Tensor, float, int], 

    ):
        # output_controlnext = {'out':controlnext_hidden_states, 'scale': scale_controlnext}
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            mid_block_additional_residual=controlnext_hidden_states,
            scale_controlnext= scale_controlnext,
            return_dict=False,
        )[0]
        return noise_pred
    


class ControlNextModel(torch.nn.Module):
    def __init__(self, controlnext):
        super().__init__()
        self.controlnext = controlnext
    
    def forward(self,sample: torch.FloatTensor, timesteps:Union[torch.Tensor, float, int]):
        output_controlnext = self.controlnext(sample, timesteps)
        return output_controlnext['out'], output_controlnext['scale'] # just  get image encoder_hidden_states of cotnrolnext
        
def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset: int,
    use_external_data_format=False,
    verbose=False,  # Thêm tham số verbose
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast("cuda"):
        if is_torch_less_than_1_11:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                enable_onnx_checker=True,
                opset_version=opset,
                verbose=verbose,  # Thêm verbose ở đây
            )
        else:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=opset,
                verbose=verbose,  # Thêm verbose ở đây
            )

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

@torch.no_grad()
def convert_models(
    controlnext_path:str,
    unet_name:str, 
    load_weight_increasement:str,
    image_model_path:str,
    ip_adapter_weight_path:str,
    output_path:str,
    opset:int=16,
    fp16: bool = False,
    lora_weight_path:str =None ,
    use_safetensors:bool= False
):
    dtype =  torch.float32

    # dtype = torch.float16 if fp16 else torch.float32
    # if fp16 and torch.cuda.is_available():
    #     device = "cuda"
    # elif fp16 and not torch.cuda.is_available():
    #     raise ValueError("float16 model export is only supported on GPUs with CUDA")
    # else:
    #     device = "cpu"
    device = 'cpu'

    #init controlnext 
    controlnext = ControlNetModel() # Hmm this is some stupid of author of the repo, i don't know why it's named like that
    #load controlnext weight
    load_safetensors(controlnext,controlnext_path)

    #init unet 
    unet =UNet2DConditionModel.from_pretrained(
        f"{unet_name}", 
        subfolder="unet", 
    )

    load_safetensors(unet, f'{load_weight_increasement}', strict=False, load_weight_increasement=False)



    #create pipeline
    # if use_safetensors is True:
    #     pipeline = StableDiffusionControlNextPipeline.from_single_file(model_path, 
    #                                                             use_safetensors=True
    #                                                             ,controlnet=controlnext,
    #                                                             unet = unet)
    # else:
    #     pipeline = StableDiffusionControlNextPipeline.from_pretrained(model_path
    #                                                             ,controlnet=controlnext,
    #                                                             unet = unet)
        

    # Load ip_adapter 
    # pipeline_ip = IPAdapterOriginal(unet,image_encoder_path = image_model_path, ip_ckpt = ip_adapter_weight_path, device = 'cuda' )

    # if lora_weight_path is not None:
    #     pipeline.load_lora_weights(lora_weight_path)
    #     pipeline.fuse_lora()




    # # TEXT ENCODER
    # num_tokens = pipeline.text_encoder.config.max_position_embeddings
    # text_hidden_size = pipeline.text_encoder.config.hidden_size
    # text_input = pipeline.tokenizer(
    #     "A sample prompt",
    #     padding="max_length",
    #     max_length=pipeline.tokenizer.model_max_length,
    #     truncation=True,
    #     return_tensors="pt",
    # )

    #UNET
     # pipeline  for convert only unet

    ip_model = IPAdapter(unet, image_encoder_path=image_model_path,ip_ckpt= ip_adapter_weight_path,device= device, num_tokens=4)
    unet = ip_model.unet


    # unet_controlnext.to(device)
    unet_in_channels = unet.in_channels
    unet_sample_size = unet.config.sample_size
    img_size = 8 * unet_sample_size
    output_path = Path(output_path)

    unet_path = output_path / "unet" / "model.onnx"
    unet_optimize = output_path / "unet_optimize" / "model.onnx"

    onnx_export(unet,
         model_args=(
                torch.randn(1, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.tensor([1.0]).to(device=device, dtype=dtype),
                torch.randn(1, 81, 768).to(device=device, dtype=dtype), 
                torch.rand(1,1280, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.tensor([2.5]).to(device=device, dtype=dtype),
                ),
                output_path=unet_path,
                ordered_input_names=[
                    "sample",
                    "timestep",
                    "encoder_hidden_state",
                    "mid_block_additional_residual",
                    "mid_block_additional_residual_scale"],
                output_names=["predict_noise"],
                 dynamic_axes={
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_state": {0: "B", 1:"2B", 2: '2B'},  # Tensor encoder hidden states
                "mid_block_additional_residual": {0: "B"},
                "predict_noise":  {0: 'b'} },
                opset=opset,
                verbose=True,
                use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split

)
    unet_opt_graph =  onnx.load(unet_path)
    onnx.save_model(
        unet_opt_graph,
       unet_optimize,  
        save_as_external_data=True, 
        all_tensors_to_one_file=True,  
        convert_attribute=False,            
        location="weights.pb",

    )

# #     #convert proj model to onnx
    device = 'cuda'
    proj = (ip_model.image_proj_model)

    image_proj_model = ImageProjModel(proj)
    proj_path = output_path / "proj" / "model.onnx"


    onnx_export(image_proj_model,
                model_args=(
                            torch.rand(1,1024).to(device= device, dtype= dtype)),
                output_path = proj_path,
                ordered_input_names=[
                    'image_embedding',
                    'clip_embedding'],
                output_names= ['image_encoder'],
                dynamic_axes={
                        'clip_embedding': {0: "batch_size", 1: "feature_dim"},
                        'image_encoder': {0: 'batch_size', 1: 'num_token', 2: 'sequences_length'}

                    } ,
                opset=opset,
                use_external_data_format=True,              
               )
    
    proj_model_path = str(proj_path.absolute().as_posix())
    proj_dir = os.path.dirname(proj_model_path)
     # optimize onnx
    shape_inference.infer_shapes_path(proj_model_path, proj_model_path)
    proj_opt_graph = optimize(onnx.load(proj_model_path), name="proj", verbose=True)
    # clean up existing tensor files
    shutil.rmtree(proj_dir)
    os.mkdir(proj_dir)
    # collate external tensor files into one
    onnx.save_model(
            proj_opt_graph,
            proj_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
    
    #convert controlnext
    controlnext_path = output_path / "controlnext" / "model.onnx"

    onnx_export(controlnext,
                model_args=(torch.rand(2,3,unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                            torch.tensor([1.0]).to(device=device, dtype=dtype),),
                output_path= controlnext_path,
                ordered_input_names=[
                    'sample',
                    'timesteps'],
                output_names=["controlnext_hidden_states", 'scale'],
                dynamic_axes={
                    'sample': {0: "B", 2: "H", 3: "W" }
                },
                opset= opset,
                use_external_data_format= True,            
                      )


    controlnext_model_path = str(controlnext_path.absolute().as_posix())
    controlnext_dir = os.path.dirname(controlnext_model_path)
     # optimize onnx
    shape_inference.infer_shapes_path(controlnext_model_path, controlnext_model_path)
    controlnext_opt_graph = optimize(onnx.load(controlnext_model_path), name="proj", verbose=True)
    # clean up existing tensor files
    shutil.rmtree(controlnext_dir)
    os.mkdir(controlnext_dir)
    # collate external tensor files into one
    onnx.save_model(
            controlnext_opt_graph,
            controlnext_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
    
if __name__ == "_main_":
    convert_models(
        controlnext_path='/home/tiennv/chaos/weight_folder/controlnet.safetensors',
        load_weight_increasement='/home/tiennv/chaos/weight_folder/unet.safetensors',
        image_model_path='/home/tiennv/chaos/weight_folder/clip_model',
        ip_adapter_weight_path='/home/tiennv/chaos/weight_folder/ip-adapter_sd15.bin',
        output_path='/home/tiennv/chaos/trash',
        opset=16,
        # fp16=True,
        lora_weight_path='/home/chaos/Documents/Chaos_project/model/sd_model/ip-adapter-faceid-plus_sd15_lora.safetensors',
        use_safetensors=True,
        unet_name= '/home/tiennv/chaos/weight_folder/unet')