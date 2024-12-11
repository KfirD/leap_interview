# leap_interview
This is my submission for the leap labs interview. It is meant to be a small library the allows users to generate adversarial images, i.e. images that appear to the human eye to fall under one category, but are labeled but which fools a particular image classifier into labeling it under a different category. 

The model which we consider is resnet18 in the torchvision library (documentation: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).

# Current State

The "proof of concept.ipynb" notebook provides a basic skeleton for attacking resnet18. Run all cells sequentially, and the final cell will show a parrot which resnet18 recognizes as a tiger.

# TODO

- Move code from jupyter notebook to python file
- Add code which automatically checks for and installs any missing libraries
- Define clear functionality for the user
- Add to documentation possible attack labels
- Give user easy access to training parameters
- Make code pep8 compliant
