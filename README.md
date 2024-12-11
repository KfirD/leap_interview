# leap_interview
This is my submission for the leap labs interview. It is meant to be a small library the allows users to generate adversarial images, i.e. images that appear to the human eye to fall under one category, but which fools a particular image classifier into labeling it under a different category. 

The model which we consider is ResNet18 in the torchvision library (documentation: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).

# Documentation

Run `run.py` as a standalone program to perform an adversarial attack on an image. Given an input image and a target label, the script generates a visually indistinguishable image that ResNet18 misclassifies as the specified target label.

#### Command-Line Arguments:
- `--image_path` (required): Specifies the path to the input image.
- `--target_label` (required): Specifies the target label that ResNet18 should classify the output image as.



See `data/labels.json` for list of allowed target labels.



As an example, from the root directory run

`python3 run.py --image_path data/grey.jpeg --target_label macaw` 



# TODO

- Give user easy access to training parameters
- Add code which automatically checks for and installs any missing libraries
- Make code pep8 compliant
