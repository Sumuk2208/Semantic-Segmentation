"""
File: cityscapes_segmentation.py
Description: Semantic segmentation pipeline using Cityscapes dataset with U-Net and PyTorch Lightning.

Dependencies:
    - torch
    - torchvision
    - albumentations
    - pytorch_lightning
    - segmentation_models_pytorch
    - matplotlib
    - PIL
    - numpy
    - torchmetrics
"""
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
dataset = Cityscapes('C:/Users/sumuk/Desktop/CSCI635/Project/data', split='train', mode='fine',
                      target_type='semantic')
ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(n_classes), colors))
def encode_segmap(mask):
    """Encodes the segmentation mask, removing unwanted classes."""
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask
def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

transform=A.Compose(
[
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=transform(image=np.array(image), mask=np.array(target))
        return transformed['image'],transformed['mask']

dataset=MyClass('C:/Users/sumuk/Desktop/CSCI635/Project/data', split='val', mode='fine',
                     target_type='semantic',transforms=transform)
img,seg= dataset[20]
print(img.shape,seg.shape)
#class labels after label correction
res=encode_segmap(seg.clone())
print(res.shape)
print(torch.unique(res))
print(len(torch.unique(res)))

res1=decode_segmap(res.clone())
from pytorch_lightning import seed_everything, LightningModule, Trainer
import multiprocessing
import torch
import torchmetrics

class OurModel(LightningModule):
    def __init__(self):
        super(OurModel, self).__init__()
        self.layer = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
        self.lr = 1e-3
        self.batch_size = 16
        self.numworker = multiprocessing.cpu_count() // 4

        self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.train_iou_metric = torchmetrics.JaccardIndex(num_classes=n_classes, task="multiclass")
        self.val_iou_metric = torchmetrics.JaccardIndex(num_classes=n_classes, task="multiclass")

        self.train_class = MyClass('C:/Users/sumuk/Desktop/CSCI635/Project/data', split='train', mode='fine',
                                   target_type='semantic', transforms=transform)
        self.val_class = MyClass('C:/Users/sumuk/Desktop/CSCI635/Project/data', split='val', mode='fine',
                                 target_type='semantic', transforms=transform)

    def process(self, image, segment):
        out = self(image)
        segment = encode_segmap(segment)  # Ensure ground truth is encoded correctly
        loss = self.criterion(out, segment.long())

        # Calculate IoU
        predictions = torch.argmax(out, dim=1)  # Convert logits to class predictions
        iou = self.train_iou_metric(predictions, segment)
        return loss, iou

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(self.train_class, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.numworker, pin_memory=True)

    def training_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def val_dataloader(self):
        return DataLoader(self.val_class, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.numworker, pin_memory=True)

    def validation_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def main():
    model = OurModel()
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='checkpoints',
                                            filename='file',save_last=True)



    trainer = Trainer(
        max_epochs=200, # Total number of epochs
        accelerator="gpu",  # Specify GPU usage
        devices="auto",  # Use all available GPUs

    )


    trainer.fit(model)
    torch.save(model.state_dict(), 'model.pth')
    test_class = MyClass('C:/Users/sumuk/Desktop/CSCI635/Project/data', split='val', mode='fine',
                         target_type='semantic',transforms=transform)
    test_loader=DataLoader(test_class, batch_size=12,
                          shuffle=False)
    model=model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            img,seg=batch
            output=model(img.cuda())
            break
    print(img.shape,seg.shape,output.shape)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    sample=6
    invimg=inv_normalize(img[sample])
    outputx=output.detach().cpu()[sample]
    encoded_mask=encode_segmap(seg[sample].clone()) #(256, 512)
    decoded_mask=decode_segmap(encoded_mask.clone())  #(256, 512)
    decoded_ouput=decode_segmap(torch.argmax(outputx,0))
    fig,ax=plt.subplots(ncols=3,figsize=(16,50),facecolor='white')
    ax[0].imshow(np.moveaxis(invimg.numpy(),0,2)) #(3,256, 512)
    #ax[1].imshow(encoded_mask,cmap='gray') #(256, 512)
    ax[1].imshow(decoded_mask) #(256, 512, 3)
    ax[2].imshow(decoded_ouput) #(256, 512, 3)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].set_title('Input Image')
    ax[1].set_title('Ground mask')
    ax[2].set_title('Predicted mask')
    plt.savefig('result.png',bbox_inches='tight')
if __name__=='__main__':
    main()
