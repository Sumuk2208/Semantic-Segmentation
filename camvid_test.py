"""
File: process_camvid_images.py
Description: Script for processing CamVid validation images using a trained segmentation model. 
             The script resizes images, applies transformations, and generates predicted segmentation masks.
"""
from city import OurModel, MyClass, transforms, transform, decode_segmap, encode_segmap
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image

# Ensure the results directory exists
os.makedirs('results_camvid', exist_ok=True)

# Load the model
model = OurModel()
model.load_state_dict(torch.load('model.pth'))  # Load saved weights
model = model.cuda()
model.eval()

# Inverse normalization
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)

# Directory containing images
image_dir = 'C:/Users/sumuk/Downloads/camvid/CamVid/val'  # Replace with the path to your image directory
output_dir = 'results_camvid'  # Directory where results will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def resize_image(image, target_size):
    """
    Resize the given image to the specified target size using LANCZOS resampling.

    Args:
        image (PIL.Image.Image): The input image.
        target_size (tuple): The desired width and height as (width, height).

    Returns:
        PIL.Image.Image: The resized image.
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


# Process each image in the directory
for image_name in os.listdir(image_dir):
    # Check if the file is an image (optional, for safety)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):

        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format

        # Resize the image to the target size expected by the model (e.g., 224x224)
        target_size = (224, 224)  # Example target size
        image = resize_image(image, target_size)

        # Transform the image
        transform_single_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_image = transform_single_image(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        # Make prediction
        with torch.no_grad():
            output = model(input_image)

            # Get the predicted output
            outputx = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
            decoded_output = decode_segmap(torch.argmax(outputx, 0))

        # Save only the predicted mask as an image
        output_image_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_predicted.png')
        plt.imsave(output_image_path, decoded_output)  # Save predicted mask

        print(f'Saved predicted mask for {image_name}')
