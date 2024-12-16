import torch
from torch.utils.data import DataLoader
from data_preparation import DeepGlobe

dataset_dir = "./data/deepglobe"
train = DeepGlobe(dataset_dir, ['train'])                       #loads only training images and masks
trainLoader = DataLoader(train, batch_size=32, shuffle=True)
test = DeepGlobe(dataset_dir, ['test','valid'])                 #loads only test / valid images from the dataset, not sure if necessary, since there are no masks...
testLoader = DataLoader(test, batch_size=32, shuffle=True)

import matplotlib.pyplot as plt
images, masks = next(iter(trainLoader))
#Look at first images
for i in range(32):
    image = images[i].permute(1, 2, 0)
    mask = masks[i]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask")
    ax[1].axis('off')
    plt.show()
