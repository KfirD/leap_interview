# leap_interview
This is my submission for the leap labs interview. It is meant to be a small library the allows users to generate adversarial images, i.e. images that appear to the human eye to fall under one category, but are labeled but which fools a particular image classifier into labeling it under a different category. 

The model which we consider is ResNet18 in the torchvision library (documentation: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).

# Documentation

Run `attack_image.py` as a standalone program to perform an adversarial attack on an image. Given an input image and a target label, the script generates a visually indistinguishable image that ResNet18 misclassifies as the specified target label.

#### Command-Line Arguments:
- `--image_path` (required): Specifies the path to the input image.
- `--save_path` (required): Specifies the path where the output image will be saved.
- `--target_label` (required): Specifies the target label that ResNet18 should classify the output image as.



# TODO

- Separate standalone program from importable library
- Add functionality that says if attack failed or succeeded
- Add code which automatically checks for and installs any missing libraries
- Define clear functionality for the user
- Add to documentation possible attack labels
- Give user easy access to training parameters
- Make code pep8 compliant
