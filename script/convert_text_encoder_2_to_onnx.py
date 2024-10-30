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
from diffusers import DiffusionPipeline



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

class TextEncoder2(torch.nn.Module):
    def __init__(self, text_encoder_2_model):
        super().__init__()
        self.text_encoder_2_model = text_encoder_2_model

    def forward(self,input_ids):
        out = self.text_encoder_2_model(input_ids, output_hidden_states = True)
        return out.text_embeds, out.last_hidden_state, out.hidden_states[32]


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



def convert_models(
    stable_diffusion_name_or_path:str, 
    output_path:str,
    opset:int=16,
    fp16: bool = False,
):
        dtype =  torch.float32
        device = 'cpu'
        pipe=  DiffusionPipeline.from_pretrained(stable_diffusion_name_or_path)
        text_encoder_model = TextEncoder2(pipe.text_encoder_2)
        output_path = Path(output_path)

        text_encoder_2 = output_path / "text_encoder_2" / "model.onnx"
        text_encoder_2_optimize = output_path / 'optimize' / 'model.onnx'
        #create folder for optimize clip
        os.makedirs(output_path / 'optimize', exist_ok= True)


        
        num_tokens = pipe.text_encoder.config.max_position_embeddings
        text_hidden_size = pipe.text_encoder.config.hidden_size
        text_input = pipe.tokenizer(
                "A sample prompt",
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        onnx_export(text_encoder_model,
                    model_args= (text_input.input_ids).to(dtype = torch.long, device = device),
                    output_path =text_encoder_2,
                    ordered_input_names= ['input_ids'],
                    output_names=["text_embeds", 'last_hidden_state','hidden_states_31'],
                    dynamic_axes={'input_ids': {0: 'Batch_size', 1: 'num_token'},
                                 'text_embeds': {0:'Batch_size', 1: 'num_token', 2: 'sequence_length'},
                                  'hidden_states_31': {0:'Batch_size', 1: 'num_token', 2: 'sequence_length'}},
                    opset=opset,
                    verbose=True,
                    use_external_data_format=True, 
                )
        text_encoder_2_opt_graph =  onnx.load(text_encoder_2)
        onnx.save_model(
            text_encoder_2_opt_graph,
            text_encoder_2_optimize,  
            save_as_external_data=True, 
            all_tensors_to_one_file=True,  
            convert_attribute=False,            
            location="weights.pb",

        )
convert_models(stable_diffusion_name_or_path= 'neta-art/neta-xl-2.0',
                      output_path='/home/tiennv/trang/text_encoder_2',
                       opset=18,
                     )

# path_image= '/home/tiennv/trang/Identities/2224.jpg'
# image =  Image.open(path_image)
# image_encoder_processor = CLIPImageProcessor()
# image_embedding = image_encoder_processor(image, return_tensors="pt").pixel_values
# image_encoder= CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder = 'models/image_encoder')
# clip_embedding = image_encoder(image_embedding,output_hidden_states=True).hidden_states[-2]
# print(clip_embedding.shape)
