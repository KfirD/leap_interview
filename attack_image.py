import argparse
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
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
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    # get index of target label
    target_idx = get_target_idx(attack_target)
    
    # initialize noise
    noise = torch.randn(image_data.shape)
    noise.requires_grad = True
    
    optimizer = torch.optim.AdamW([noise], lr=0.01)
    
    l1_lambda = 0.1
    
    # train noise
    for step in range(500):
        optimizer.zero_grad()
        logits = model(image_data + noise)
        
        l1_norm = noise.abs().sum()
        
        loss = F.cross_entropy(logits, target_idx) + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
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
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    attack_data = image_to_tensor(attack_image)
    
    with torch.no_grad():
        output = model(attack_data)
    _, predicted = torch.max(output, 1)
    predicted_label = labels[predicted.item()]
    
    print(f'{predicted_label=}')
    print(f'{attack_target=}')
    
    return predicted_label == attack_target
