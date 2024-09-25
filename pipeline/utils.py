from pathlib import Path
import onnxruntime as ort
import json
from typing import Any, Union, List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class StableDiffusionControlNextLoader:
    folder_model:str

    def __post_init__(self):
        self.folder_model  = Path(self.folder_model)
        self.controlnext = self.folder_model / "controlnext" / 'model.onnx'
        self.proj = self.folder_model / "proj" / 'model.onnx'
        self.scheduler = self.folder_model / 'scheduler'
        self.text_encoder = self.folder_model / "text_encoder" / "model.onnx"
        self.tokenizer = self.folder_model / "tokenizer"
        self.unet_optime = self.folder_model / "unet_optimize" / 'model.onnx'
        self.vae_decoder = self.folder_model / "vae_decoder" / 'model.onnx'
        self.vae_encoder = self.folder_model / "vae_encoder" / 'model.onnx'
        self.face_analysis = self.folder_model / "face_analysis"

class OnnxExecute:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.inputs = None
        self.device = None
        self.session = None

    def create_onnx_session(self):
        # Create the ONNX inference session based on the specified device
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)

    def get_metadata_onnx(self):
        self.create_onnx_session()
        input_meta = self.session.get_inputs()

        # Print the input metadata
        print("Input metadata:")
        for i, input in enumerate(input_meta):
            print(f"Input {i}:")
            print(f"  Name: {input.name}")
            print(f"  Shape: {input.shape}")

        # Optionally, get and print the model's output metadata
        output_meta = self.session.get_outputs()
        print("\nOutput metadata:")
        for i, output in enumerate(output_meta):
            print(f"Output {i}:")
            print(f"  Name: {output.name}")
            print(f"  Shape: {output.shape}")


    def execute(self):
        if self.session is None:
            raise RuntimeError("ONNX session is not created. Call 'create_onnx_session()' first.")
        
        # Execute the model with provided inputs
        outputs = self.session.run(None, self.inputs)
        return outputs

    def __call__(self, inputs: Dict, device: str) -> Any:

        self.inputs = inputs
        self.device = device
        self.create_onnx_session()
        return self.execute()
    


execute = OnnxExecute('/home/chaos/Documents/Chaos_project/model/sd_controlnext_fp16_onnx/unet_optimize/model.onnx')
execute.get_metadata_onnx()