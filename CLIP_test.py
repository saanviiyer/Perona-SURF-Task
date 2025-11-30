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