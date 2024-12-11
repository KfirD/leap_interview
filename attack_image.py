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

# ================================
# main functionality
# ================================

def attack_image(
    image_path: str,
    save_path: str,
    attack_target: str 
    ):
    
    # proccess image for resnet18
    image_data = proccess_image(image_path)    
    
    # calculate noise which minimizes loss with respect to target output
    attack_noise = calculate_noise(image_data, attack_target)
    
    # generate and save output image
    generate_and_save_image(image_data, 0, "hello")
    generate_and_save_image(image_data, attack_noise, save_path)

# ================================
# helper functions
# ================================

def proccess_image(image_path: str) -> torch.Tensor:
    
    # Function to prepare images for input
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load an image
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor

# TODO, make labels an attribute in the class so they're not a floating global variable

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(url).json()

# return idx of target
def get_target_idx(attack_target: str) -> int:
    idx = labels.index(attack_target)
    out = torch.tensor(idx).reshape(1)
    return out 

# todo: make training config and give user easier control over it

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
    
    l1_lambda = 0.1
    
    # train noise
    for step in range(100):
        optimizer.zero_grad()
        logits = model(image_data + noise)
        
        l1_norm = noise.abs().sum()
        
        loss = F.cross_entropy(logits, target_idx) + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f'step: {step}, loss: {loss}')
                        
    return noise

# TODO clean this up
def predict_label(model, input_tensor):
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(url).json()
    
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_label = labels[predicted.item()]
    
    return predicted_label
    
# Reverse the preprocessing transformations
def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    # Remove the batch dimension if it exists
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)
    
    # # Unnormalize the image
    # unnormalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    # )
    # unnormalized_tensor = unnormalize(tensor)
    
    # Clip the values to be between 0 and 1
    unnormalized_tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(unnormalized_tensor)
    
    return image

# TODO: add show image as an option from command line
def generate_and_save_image(
    image_data: torch.Tensor, 
    attack_noise: torch.Tensor, 
    save_path: str):

    output_image = tensor_to_image(image_data + attack_noise)
    output_image.show()  # Display the image


def main():
    # Parse command-line arguments for attack_image
    parser = argparse.ArgumentParser(description="Given an image and a target label, generates image indistinguishable to human eye which resnet18 classifies as the target label.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save output image.")
    parser.add_argument('--target_label', type=str, required=True, help="Path to save output image.")
    args = parser.parse_args()

    # Call attack_image with parsed arguments
    attack_image(args.image_path, args.save_path, args.target_label)

    # TODO report if attack failed

if __name__ == "__main__":
    main()
