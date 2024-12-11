import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image
import json
import requests
import numpy as np

from utils.image_processing import *
from utils.image_display import *


# TODO remove global variable model 
# ================================
# main functionality
# ================================

def generate_attack_image(
    input_image: Image,
    attack_target: str, 
    ):
    
    # proccess image for resnet18
    image_data = image_to_tensor(input_image)    
    
    # calculate noise which minimizes loss with respect to target output
    attack_noise = calculate_noise(image_data, attack_target)
    
    # generate
    attack_image = tensor_to_image(image_data + attack_noise)
    
    return attack_image

# ================================
# adversarial noise calculation
# ================================

# TODO get labels locally
# TODO move training parameters to cfg class and make commandline args

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(url).json()

# return idx of target
def get_target_idx(attack_target: str) -> int:
    idx = labels.index(attack_target)
    out = torch.tensor(idx).reshape(1)
    return out 

def calculate_noise(image_data: torch.Tensor, attack_target: str) -> torch.Tensor:
    
    # intialize model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # get index of target label
    target_idx = get_target_idx(attack_target)
    
    # initialize noise
    noise = torch.randn(image_data.shape)
    noise.requires_grad = True
    
    optimizer = torch.optim.AdamW([noise], lr=0.01)
    
    l1_lambda = 0.01
    
    # train noise
    for step in range(200):
        optimizer.zero_grad()
        logits = model(image_data + noise)
        
        l1_norm = noise.abs().sum()
        
        loss = F.cross_entropy(logits, target_idx) + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f'step: {step}, loss: {loss}')
                        
    return noise

# ================================
# check if image was fooled
# ================================

# TODO call json labels locally
# TODO make model variable passed in
def attack_status(attack_image: Image, attack_target: str) -> bool:
    
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(url).json()
    
    model = models.resnet18(pretrained=True)
    model.eval()
    
    attack_data = image_to_tensor(attack_image)
    
    with torch.no_grad():
        output = model(attack_data)
    _, predicted = torch.max(output, 1)
    predicted_label = labels[predicted.item()]
    
    print(f'{predicted_label=}')
    print(f'{attack_target=}')
    
    return predicted_label == attack_target
    


# TODO: fix targel_label attack_target inconsistancy
def main():
    # Parse command-line arguments for attack_image
    parser = argparse.ArgumentParser(description="Given an image and a target label, generates image indistinguishable to human eye which resnet18 classifies as the target label.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--target_label', type=str, required=True, help="Path to save output image.")
    
    args = parser.parse_args()

    # load input image
    input_image = Image.open(args.image_path)
    processed_input_image = standardize_image(input_image)
    
    # generate attacked image
    attack_image = generate_attack_image(input_image, args.target_label)

    # show original and final image
    show_images_side_by_side_with_labels(processed_input_image, "standardized input image", attack_image, "attack image")
    
    print(f'attack succeeded? {attack_status(attack_image, args.target_label)}')

if __name__ == "__main__":
    main()
