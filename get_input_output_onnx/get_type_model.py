# import onnx

# def check_onnx_model_io(model_path):
#     # Load ONNX model
#     model = onnx.load(model_path)
    
#     # Get the graph from the model
#     graph = model.graph

#     # Get input names and types
#     print("Inputs:")
#     for input_tensor in graph.input:
#         input_name = input_tensor.name
#         input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
#         input_type = input_tensor.type.tensor_type.elem_type
#         print(f"Name: {input_name}, Shape: {input_shape}, Type: {onnx.TensorProto.DataType.Name(input_type)}")

#     # Get output names and types
#     print("\nOutputs:")
#     for output_tensor in graph.output:
#         output_name = output_tensor.name
#         output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
#         output_type = output_tensor.type.tensor_type.elem_type
#         print(f"Name: {output_name}, Shape: {output_shape}, Type: {onnx.TensorProto.DataType.Name(output_type)}")

# # Example usage:
# check_onnx_model_io('/home/chaos/Documents/Chaos_project/model/yolov5/final0973.onnx')

# import cv2
# import torch
# from PIL import Image

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'custom', '/home/chaos/Documents/Chaos_project/model/yolov5/final0973.onnx') 
# image_path = '/home/chaos/Documents/Chaos_project/project/craw_smart_traffic/data/dinhbolinh_bach_dang/5a8253bc5058170011f6eac1_20241002_082155.jpg'
# img = Image.open(image_path)  # PIL image

# img = img.resize((640,640))

# # Inference
# results = model(img, size=640)  # includes NMS

# # Results
# results.print()  # print results to screen
# results.show()  # display results
# results.save()  # save as results1.jpg, results2.jpg... etc.

# # Data
# print('\n', results.xyxy[0])  # print img1 predictions

import torch
from ultralytics import YOLO

# Load the model
model_path = '/home/chaos/Documents/Chaos_project/model/yolov5/final0973.onnx'
model = YOLO(model_path, task='detect')  # Load the ONNX model

# # # Specify the image path
image_path = '/home/chaos/Documents/Chaos_project/project/craw_smart_traffic/data/dinhbolinh_bach_dang/5a8253bc5058170011f6eac1_20241002_082155.jpg'

# # # Perform prediction
results = model.predict(image_path)

# # # Display results
# # results.show()  # Display the results
# # print(results)   # Print detailed results