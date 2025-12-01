import torch
from torch.utils.data import DataLoader, random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration

from PIL import Image
from tqdm import tqdm
import wandb

"""
OBJECTIVES: 

"""


wandb.login()
wandb.init(
    project="LLaVA-Caltech101-ZeroShot",
    config={
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "task": "Zero-Shot Classification",
    }
)

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(model_id)

processor.image_processor.size = {"shortest_edge": 336}
processor.image_processor.crop_size = {"height": 336, "width": 336}

pil_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img)
])

full_dataset = datasets.Caltech101(
    root="./data",
    download=True,
    transform=pil_transforms  
)
class_names = full_dataset.categories

torch.manual_seed(42) # same "randomization" every time
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

print(f"loaded {len(test_set)} test images")

num_correct = 0
num_samples = 0

with torch.no_grad():

    # taking too long to run, just limited it to 300
    num_test_images = 300
    for i in tqdm(range(num_test_images), desc="Evaluating LLaVA"):

        pil_image, label_int = test_set[i]

        label_name = class_names[label_int]

        prompt = f"USER: <image>\nWhat object is in this image? ASSISTANT:"

        inputs = processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(model.device, dtype=torch.float16)

        generate_ids = model.generate(**inputs, max_new_tokens=30)

        output_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True
        )[0]

        try:
            answer = output_text.split("ASSISTANT:")[-1].strip().lower()
        except Exception:
            answer = ""

        if label_name.lower() in answer:
            num_correct += 1
        num_samples += 1

        # test output: log a few examples
        if num_samples % 100 == 0:
            print(f"\n--- Example {num_samples} ---")
            print(f"Prompt: What object is in this image?")
            print(f"Ground Truth: {label_name}")

accuracy = (num_correct / num_samples) * 100
print(f"LLaVA Accuracy on {num_samples} images: {accuracy:.2f}%")

wandb.log({
    "llava_zero_shot_accuracy": accuracy,
    "total_correct": num_correct,
    "total_samples": num_samples
})

wandb.finish()


# BONUS!!!

# make PCA (dimensionality reduction technique) plot of LLaVA intermediate activations for images from 2 distinct classes in dataset
# make 5 plots for 5 intermediate layers
# original dimensions of embeddings
# where did i find LLaVA embeddings, why?
# difference between different intermediate layers?

