# Semantic Segmentation

## Description
This project focuses on training a UNet model with a ResNet encoder on the Cityscapes dataset. The dataset contains urban street scenes with multiple classes, which will be used to perform semantic segmentation tasks, helping to identify and label each pixel in the image based on predefined classes.

## Installation Instructions
Install the required libraries by running the following command in your Python environment:

```bash
pip install torchvision albumentations matplotlib pytorch-lightning segmentation-models-pytorch
```
Project Structure
To run this project, you will need to download the Cityscapes dataset from Cityscapes Dataset.

The required data structure is as follows:
root/ │ └── data/ ├── leftImg8bit/ │ ├── train/ │ ├── test/ │ └── val/ │ └── GTFine/ ├── train/ ├── test/ └── val/
