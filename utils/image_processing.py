import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image

def image_to_tensor(image: Image) -> torch.Tensor:
    
    # Function to prepare images for input
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load an image
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor
    
# Reverse the preprocessing transformations
def tensor_to_image(tensor: torch.Tensor) -> Image:
    # Remove the batch dimension if it exists
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)
    
    # Clip the values to be between 0 and 1
    unnormalized_tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(unnormalized_tensor)
    
    return image

def standardize_image(image: Image) -> Image:
    return tensor_to_image(image_to_tensor(image))
