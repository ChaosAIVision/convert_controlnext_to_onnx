import argparse
import os
import shutil
from pathlib import Path
from PIL import Image
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from packaging import version
from polygraphy.backend.onnx.loader import fold_constants
from torch.onnx import export

from typing import Union, Optional, Tuple
from diffusers import AutoPipelineForText2Image
from repo_ipadapter.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from repo_controlnext.controlnext_test.models.controlnet import ControlNetModel
from repo_controlnext.controlnext_test.models.pipeline_controlnext import StableDiffusionControlNextPipeline
from safetensors.torch import load_file
from repo_diffusers.src.diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer



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


class CLIPVisionProj(torch.nn.Module):
    def __init__(self, clip_model) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image_embedding):
        result = self.clip_model(image_embedding,return_dict = False)
        return result[0]
    
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

def convert_models(
    image_model_path:str,
    image_path:str, 
    output_path:str,
    opset:int=16,
    fp16: bool = False,
):
        dtype =  torch.float32
        device = 'cpu'
        image = Image.open(image_path)
        image_encoder_processor = CLIPImageProcessor()
        image_embedding = image_encoder_processor(image, return_tensors="pt").pixel_values
        clip_model= CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder = 'models/image_encoder')
        image_encoder = CLIPVisionProj(clip_model).to(device=device)
        output_path = Path(output_path)

        clip_path = output_path / "clip_vision_proj" / "model.onnx"
        clip_optimize = output_path / 'clip_vision_proj' / 'optimize' / 'model.onnx'
        #create folder for optimize clip
        os.makedirs(output_path / 'optimize', exist_ok= True)
        onnx_export(image_encoder,
                    model_args= (image_embedding).to(dtype = torch.float32, device = device),
                    output_path =clip_path,
                    ordered_input_names= ['image_embedding'],
                    output_names=["image_encoder"],
                    dynamic_axes={'image_embedding': {0: 'Batch_size',1: 'channel', 2: 'height', 3:'weidth'},
                                 'image_encoder': {0:'Batch_size', 1: 'sequence_length'} },
                    opset=opset,
                    verbose=True,
                    use_external_data_format=True, 
                )
        clip_opt_graph =  onnx.load(clip_optimize)
        onnx.save_model(
            clip_opt_graph,
            clip_optimize,  
            save_as_external_data=True, 
            all_tensors_to_one_file=True,  
            convert_attribute=False,            
            location="weights.pb",

        )

if __name__ == "_main_":
    convert_models(image_model_path= 'trash',
                    image_path= '/home/tiennv/trang/Identities/2224.jpg',
                      output_path='/home/tiennv/chaos/output',
                       opset=16,
                     )

# path_image= '/home/tiennv/trang/Identities/2224.jpg'
# image =  Image.open(path_image)
# image_encoder_processor = CLIPImageProcessor()
# image_embedding = image_encoder_processor(image, return_tensors="pt").pixel_values
# image_encoder= CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder = 'models/image_encoder')
# clip_embedding = image_encoder(image_embedding,output_hidden_states=True).hidden_states[-2]
# print(clip_embedding.shape)
