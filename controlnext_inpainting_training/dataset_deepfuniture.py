import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import pandas
from transformers import CLIPImageProcessor
from PIL import Image, ImageDraw

class Deepfurniture_Dataset(Dataset):
    def __init__(self, data, input_type, image_size = 512, proportion_empty_prompts=0.05):
        self.data = pandas.read_csv(data)
        self.image_size = image_size
        self.proportion_empty_prompts = proportion_empty_prompts
        self.input_type = input_type
        self.mask_tranfomrs = transforms.Compose([transforms.ToTensor(), 
        transforms.Resize((self.image_size, self.image_size))
        ])
        self.clip_image_embedding = CLIPImageProcessor()

        self.image_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5], [0.5])
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.data)
    
    def get_bboxes(self, bbox_string):
        processed_bboxes = list(map(int, bbox_string.split(',')))
        return processed_bboxes
    
    def make_data(self, image_path, bbox):
        pixel_values = Image.open(image_path).convert("RGB")        
        xmin, ymin, xmax, ymax = bbox
        ipadapter_images = pixel_values.crop((xmin, ymin, xmax, ymax))     
        conditioning_pixel_values = Image.new("L", pixel_values.size, 0)  
        draw = ImageDraw.Draw(conditioning_pixel_values)
        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)  
        original_values = pixel_values.copy()
        draw_original = ImageDraw.Draw(original_values)
        draw_original.rectangle([xmin, ymin, xmax, ymax], fill=(255, 255, 255))  
        mask_values = conditioning_pixel_values.copy()

        return {
            "pixel_values": pixel_values,
            "ipadapter_images": ipadapter_images,
            "conditioning_pixel_values": conditioning_pixel_values,
            "original_values": original_values,
            "mask_values": mask_values,
        }
        
    def __getitem__(self, idx):
        # Get the current item data
        item = self.data.iloc[idx]
        image_path = item['image_path']
        bbox = self.get_bboxes(item['bbox'])  # Convert bbox string to list of coordinates
        
        # Generate the required images using make_data
        data = self.make_data(image_path, bbox)
        
        # Apply transformations
        pixel_values = self.image_transforms(data["pixel_values"])
        conditioning_pixel_values = self.conditioning_image_transforms(data["conditioning_pixel_values"])
        original_values = self.image_transforms(data["original_values"])

        # Mask values depending on input_type
        if self.input_type == "merge":
            mask_values = self.mask_tranfomrs(data["mask_values"])
        else:
            mask_values = self.image_transforms(data["original_values"])

        # Process ipadapter_images with CLIP image embedding and convert to tensor
        ipadapter_images = (self.clip_image_embedding(data["ipadapter_images"], return_tensors='pt').pixel_values)[0]

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "original_values": original_values,
            "mask_values": mask_values,
            "ipadapter_images": ipadapter_images,
        }
        


