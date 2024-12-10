from city import OurModel,MyClass,transforms,transform,decode_segmap,encode_segmap,DataLoader
import os
from matplotlib import pyplot as plt
import numpy as np
import torch

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Load test data
test_class = MyClass('C:/Users/sumuk/Desktop/CSCI635/Project/data', split='val', mode='fine',
                     target_type='semantic', transforms=transform)
test_loader = DataLoader(test_class, batch_size=12, shuffle=False)

# Load model
model = OurModel()
model.load_state_dict(torch.load('model.pth'))  # Load saved weights
model = model.cuda()
model.eval()

# Inverse normalization
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)

# Visualize multiple samples
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        img, seg = batch
        output = model(img.cuda())

        for sample in range(len(img)):
            invimg = inv_normalize(img[sample].cpu())
            outputx = output.detach().cpu()[sample]
            encoded_mask = encode_segmap(seg[sample].clone())
            decoded_mask = decode_segmap(encoded_mask.clone())
            decoded_output = decode_segmap(torch.argmax(outputx, 0))

            fig, ax = plt.subplots(ncols=3, figsize=(16, 10), facecolor='white')
            ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2))
            ax[1].imshow(decoded_mask)
            ax[2].imshow(decoded_output)

            for a in ax:
                a.axis('off')

            ax[0].set_title('Input Image')
            ax[1].set_title('Ground Mask')
            ax[2].set_title('Predicted Mask')

            # Save each sample as an image
            plt.savefig(f'results/result_batch_{batch_idx}_sample_{sample}.png', bbox_inches='tight')
            plt.close(fig)  # Close the figure to save memory

        # Limit visualization to first 3 batches for brevity
        if batch_idx == 2:
            break
