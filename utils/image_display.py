from PIL import Image, ImageDraw, ImageFont

def show_images_side_by_side_with_labels(img1, label1, img2, label2):
    # Load images
    img1_width, img1_height = img1.size
    img2_width, img2_height = img2.size
    
    # Determine label height
    label_height = 50

    # Create a blank canvas with space for labels
    total_width = img1_width + img2_width
    total_height = max(img1_height, img2_height) + label_height
    combined_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # Draw images onto the canvas
    combined_image.paste(img1, (0, label_height))
    combined_image.paste(img2, (img1_width, label_height))

    # Add labels
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.load_default()  # You can specify a custom font here if needed
    draw.text((img1_width // 2, 10), label1, fill="black", anchor="mm", font=font)
    draw.text((img1_width + img2_width // 2, 10), label2, fill="black", anchor="mm", font=font)

    # Show the combined image
    combined_image.show()

