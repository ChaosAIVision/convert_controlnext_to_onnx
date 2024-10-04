import numpy as np
import onnxruntime as ort
import torch
import time  # Import thư viện time để đo thời gian

# Khởi tạo dummy input theo metadata của mô hình
B = 1  # Batch size
H, W = 64, 64  # Height và Width của ảnh
dtype = np.float32

# Dummy input 0: 'sample' với shape ['2B', 4, 'H', 'W']
sample = np.random.randn(1, 4, H, W).astype(dtype)

# Dummy input 1: 'timesteps' với shape [1]
timesteps = np.array([1], dtype=dtype)

# Dummy input 2: 'encoder_hidden_states' với shape ['B', '2B', '2B']
encoder_hidden_states = np.random.randn(B, 77, 768).astype(dtype)

# Dummy input 3: 'controlnext_hidden_states' với shape ['B', 1280, 8, 8]
controlnext_hidden_states = np.random.randn(B, 1280, 8, 8).astype(dtype)
scale = np.array([1], dtype=dtype)
# Gói tất cả input vào từ điển để phù hợp với mô hình ONNX
inputs = {
    'sample': sample,
    'timesteps': timesteps,
    'encoder_hidden_states': encoder_hidden_states,
    'controlnext_hidden_states': controlnext_hidden_states,
    "scale_controlnext": scale
}

# Load mô hình ONNX với GPU
onnx_model_path = '/home/chaos/Documents/Chaos_project/model/sd_controlnext_fp16_onnx/unet_optimize/unet_optimize/model.onnx'
providers = ['CPUExecutionProvider']  # Sử dụng GPU
session = ort.InferenceSession(onnx_model_path, providers=providers)

# Đo thời gian inference
start_time = time.time()  # Bắt đầu đo thời gian
outputs = session.run(None, inputs)  # Chạy inference
end_time = time.time()  # Kết thúc đo thời gian

# In output đầu tiên ('noise_pred')
print("Output (noise_pred) shape:", outputs[0])

# Tính toán và in thời gian inference
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# from torchinfo import summary
# import torch
# from repo_controlnext.controlnext_test.models.controlnet import ControlNetModel
# from .convert_controlnext_to_onnx import ControlNextModel
# controlnet = ControlNetModel()
# controlnext = ControlNextModel(controlnet)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# controlnet.to(device)

# image = torch.randn((2, 4, 64, 64)).to(device)
# timestep = torch.randn(2).to(device)
# encoder_hidden_states = torch.randn((2, 77, 768)).to(device)
# controlnet_cond = torch.randn((2, 3, 512, 512)).to(device)

# summary(controlnext.forward(), input_data=(image, timestep, encoder_hidden_states, controlnet_cond))