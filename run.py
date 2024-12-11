import argparse
from PIL import Image


from utils.image_processing import *
from utils.image_display import *
from attack_image import *

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
