# Semantic Segmentation

## Description
This project focuses on training a UNet model with a ResNet encoder on the Cityscapes dataset. The dataset contains urban street scenes with multiple classes, which will be used to perform semantic segmentation tasks, helping to identify and label each pixel in the image based on predefined classes.
### Python Libraries

- **torch**: The core package for deep learning using PyTorch, which provides tensor computations and deep learning functionality.
- **torchvision**: Provides vision-related utilities, including datasets (such as Cityscapes) and transforms.
- **pytorch_lightning**: Simplifies the process of training and organizing PyTorch code, with built-in support for callbacks such as EarlyStopping and ModelCheckpoint.
- **albumentations**: A fast image augmentation library for training machine learning models with advanced image transformations.
- **segmentation_models_pytorch (smp)**: A library for semantic segmentation models, providing architectures like U-Net, FPN, and DeepLabV3.
- **matplotlib**: A library for generating plots, used here for visualizing data and model predictions.
- **PIL (Python Imaging Library)**: Provides image processing functionalities, including image opening, saving, and transforming.
- **numpy**: A fundamental package for scientific computing in Python, used here for handling arrays.
### Code Imports

```python
from torchvision.datasets import Cityscapes
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import segmentation_models_pytorch as smp
```
Project Structure
To run this project, you will need to download the Cityscapes dataset from Cityscapes Dataset.

The required data structure is as follows:
root/
│
└── data/
    ├── leftImg8bit/
    │   ├── train/
    │   ├── test/
    │   └── val/
    │
    └── GTFine/
        ├── train/
        ├── test/
        └── val/
## Usage Instructions
Run the following files in order to train the model and generate the results:

city.py

test.py

camvid_test.py

generate_image.py

create_video.py

merge_video.py

