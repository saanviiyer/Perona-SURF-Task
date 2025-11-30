import requests
import torch

from PIL import Image

from torchvision.datasets import Caltech101

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from transformers import AutoProcessor, AutoModel, pipeline, CLIPModel, CLIPProcessor

from tqdm import tqdm
import wandb

# lightning imports
import os
from torch import optim, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


# preprocessing CLIP images

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
target_size = processor.image_processor.crop_size["height"]
print(target_size) # 224

img_transforms = transforms.Compose([
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = processor.image_processor.image_mean,
        std = processor.image_processor.image_std
    )
])

dataset = Caltech101(root = './data', download = True, transform = img_transforms)

# check # categories
# print(dataset.__len__())
num_categories = len(dataset.categories)
print(num_categories) # should be 101

# split into training/validation/testing data (70, 15, 15)
size = len(dataset)
training_size = int(0.7 * size)
validation_size = int(0.15 * size)
testing_size = size - training_size - validation_size

# random split
train_set, val_set, test_set = random_split(dataset, [training_size, validation_size, testing_size])

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

# use lightning module, define class
class LightningCLIP(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def training_step(self, batch, batch_index):
        x = batch[0]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        return loss

def prepare_batch(batch_data):
    images = []
    texts = []
    labels = []

    for image, label in batch_data:
        images.append(image)
        category = dataset.categories[label]
        texts.append(f"a photo of a {category.replace('_', ' ')}")
        labels.append(label)
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return inputs, torch.tensor(labels)

train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
val_loader = DataLoader(train_set, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(),__init__()
        self.fulCon1 = nn.Linear(input_size, 50)
        self.fulCon2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fulCon1(x))
        x = self.fulCon2(x)
        return x
    
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3
model = NN(input_size=target_size, num_classes=num_categories)

# Set Up