import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import pandas
from transformers import CLIPImageProcessor

class StableDiffusionDataset(Dataset):
    def __init__(self, data, tokenizer,input_type,  image_size=512, proportion_empty_prompts=0.05):
        self.data = pandas.read_csv(data)
        self.tokenizer = tokenizer
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

    def tokenize_captions(self, captions, is_train=True):
        tokenized_captions = []
        for caption in captions:
            if random.random() < self.proportion_empty_prompts:
                tokenized_captions.append("PBH")
            elif isinstance(caption, str):
                tokenized_captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                tokenized_captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column should contain either strings or lists of strings.")
        
        inputs = self.tokenizer(
            tokenized_captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        pixel_values = Image.open(item['pixel_values']).convert("RGB")
        pixel_values = self.image_transforms(pixel_values)

        conditioning_pixel_values = Image.open(item['conditioning_pixel_values']).convert("RGB")
        conditioning_pixel_values = self.conditioning_image_transforms(conditioning_pixel_values)

        original_values = Image.open(item['original_values']).convert("RGB")
        original_values = self.image_transforms(original_values)

        if self.input_type == "merge":
            mask_values = Image.open(item['mask_values']).convert("L")
            mask_values = self.mask_tranfomrs(mask_values)
        else:
            mask_values = Image.open(item['mask_values']).convert("RGB")
            mask_values = self.image_transforms(mask_values)

        ipadapter_images = Image.open(item['ipadapter_images']).convert("RGB")
        ipadapter_images = (self.clip_image_embedding(ipadapter_images,return_tensors = 'pt').pixel_values)[0]

        text_prompt = item['text_prompt']
        input_ids = self.tokenize_captions([text_prompt])

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "original_values": original_values,
            "mask_values": mask_values,
            "ipadapter_images": ipadapter_images,
            "input_ids": input_ids 
        }





# from transformers import  CLIPTokenizer
# tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
# dataset = StableDiffusionDataset('/home/tiennv/chaos/data.csv',tokenizer,'merge' )
# # accelerator = Accelerator() 
# # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1,collate_fn=collate_fn)
# # print(len(dataloader))

# masks = (dataset.__getitem__(0)['mask_values'])
# Giả sử masks là một danh sách chứa các giá trị của mask
# print(masks.shape)
# # Kiểm tra xem có giá trị 0 trong danh sách masks không
# contains_zero = 0 in masks

# if contains_zero:
#     print("Có giá trị 0 trong masks.")
# else:
#     print("Không có giá trị 0 trong masks.")
