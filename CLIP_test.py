import requests
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModel, pipeline

from torchvision.datasets import Caltech101
from torchvision import transforms

# Preprocessing Images with CLIP

