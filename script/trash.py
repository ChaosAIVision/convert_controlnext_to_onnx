import numpy as np
import onnxruntime as ort
import torch

# Khởi tạo dummy input theo metadata của mô hình
B = 1  # Batch size
H, W = 64, 64  # Height và Width của ảnh

# Dummy input 0: 'sample' với shape ['2B', 4, 'H', 'W']
sample = np.random.randn(1, 4, H, W).astype(np.float16)

# Dummy input 1: 'timesteps' với shape [1]
timesteps = np.array([1], dtype=np.float16)

# Dummy input 2: 'encoder_hidden_states' với shape ['B', '2B', '2B']
encoder_hidden_states = np.random.randn(B, 77, 768).astype(np.float16)

# Dummy input 3: 'controlnext_hidden_states' với shape ['B', 1280, 8, 8]
controlnext_hidden_states = np.random.randn(B, 1280, 8, 8).astype(np.float16)

# Gói tất cả input vào từ điển để phù hợp với mô hình ONNX
inputs = {
    'sample': sample,
    'timesteps': timesteps,
    'encoder_hidden_states': encoder_hidden_states,
    'controlnext_hidden_states': controlnext_hidden_states,
}

# Load mô hình ONNX với GPU
onnx_model_path = '/home/chaos/Documents/Chaos_project/model/sd_controlnext_fp16_onnx/unet_optimize/model.onnx'
providers = ['CUDAExecutionProvider']  # Sử dụng GPU
session = ort.InferenceSession(onnx_model_path, providers=providers)

# Chạy suy luận (inference)
outputs = session.run(None, inputs)

# In output đầu tiên ('noise_pred')
print("Output (noise_pred) shape:", outputs[0])
