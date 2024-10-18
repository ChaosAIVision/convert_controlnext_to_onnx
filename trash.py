# # Re-run the code to test locally

# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt

# # Placeholder VAE encoder and decoder functions
# def vae_encoder(image):
#     """Simulate the VAE encoder which downsamples image from (1, 3, 512, 512) to (1, 4, 64, 64)."""
#     return F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)

# def vae_decoder(latents):
#     """Simulate the VAE decoder which upsamples latents from (1, 4, 64, 64) to (1, 3, 512, 512)."""
#     return F.interpolate(latents, size=(512, 512), mode='bilinear', align_corners=False)

# # Function to visualize image tensors
# def visualize_image(image, title="Image"):
#     image = image.squeeze().cpu().numpy()
#     image = np.transpose(image, (1, 2, 0))  # Convert to HWC format for visualization
#     plt.imshow(np.clip(image, 0, 1))  # Clip values to [0, 1] range
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

# # Function to add noise only in masked regions after encoding the image
# def encode_and_add_noise(image, mask, timesteps, device, dtype=torch.float32):
#     """
#     Encode the image using a VAE encoder, add noise only to the masked regions in latent space, and then decode it back.
    
#     Args:
#         image: Original image tensor (1, 3, 512, 512).
#         mask: Mask image tensor (1, 1, 512, 512) with 1 indicating regions to add noise.
#         timesteps: Placeholder timesteps used to determine the noise level.
#         device: Device to perform computation ('cuda' or 'cpu').
#         dtype: Data type for tensors.
    
#     Returns:
#         decoded_image: The image with noise added only to the masked regions.
#     """
#     # Move input image and mask to the device and type
#     image = image.to(device=device, dtype=dtype)
#     mask = mask.to(device=device, dtype=dtype)

#     # Encode the image using the VAE encoder
#     latents = vae_encoder(image)  # Output latent space (1, 4, 64, 64)

#     # Resize the mask to match the latent space dimensions
#     mask = F.interpolate(mask, size=(latents.shape[2], latents.shape[3]))  # Resize to (1, 1, 64, 64)
    
#     # Create random noise in the latent space
#     noise = torch.randn_like(latents, device=device, dtype=dtype)  # Noise has shape (1, 4, 64, 64)
#     print(noise.shape, 'day la noise')
#     print(mask.shape, 'day la mask')
#     print(latents.shape , 'day la latents')
    
#     # Add noise only to the masked regions
#     noised_latents = mask * noise + (1 - mask) * latents  # Noise in masked area, original latent in unmasked area
#     print(noised_latents.shape , 'day la noised_latents')

#     # Decode the latents back to image space
#     decoded_image = vae_decoder(noised_latents)
    
#     return decoded_image

# # Simulate input image and mask for testing
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# image = torch.rand((1, 4, 512, 512), device=device)  # Original image (1, 3, 512, 512)
# mask = torch.zeros((1, 1, 512, 512), device=device)  # Mask image (1, 1, 512, 512)
# mask[:, :, 150:350, 150:350] = 1  # Mask a central square region

# # Placeholder timesteps (not used in this simplified example)
# timesteps = torch.tensor([10], device=device)

# # Add noise only in masked region
# decoded_image = encode_and_add_noise(image, mask, timesteps, device)

# # # Visualize the original image, mask, and the result
# # visualize_image(image, title="Original Image")
# # # visualize_image(mask, title="Mask Image")
# # visualize_image(decoded_image, title="Image with Noise in Masked Region")

from PIL import Image
from torchvision import transforms

# 1. Mở ảnh PIL
pil_image = Image.open('/Users/chaos/Downloads/COCO_train2014_000000122688.jpg')

# 2. Chuyển ảnh sang grayscale (1 kênh duy nhất)
pil_gray_image = pil_image.convert('L')  # 'L' là mode cho grayscale

# 3. Chuyển đổi từ ảnh PIL sang Tensor
transform = transforms.ToTensor()
gray_image_tensor = transform(pil_gray_image).unsqueeze(0)  # unsqueeze(0) để thêm batch dimension

# Kết quả là tensor với shape [1, 1, 512, 512]
print(gray_image_tensor.shape)

