from PIL import Image


def adjust_brightness(image_path, output_path, dark_threshold=128, bright_threshold=128):
    # Open the image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB mode
    pixels = img.load()  # Load pixel data

    # Get image dimensions
    width, height = img.size

    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]

            # Calculate the brightness of the pixel
            brightness = (r + g + b) // 3  # Average to determine brightness

            # Brighten dark content
            if brightness < dark_threshold:
                # Increase brightness (scale factor can be adjusted)
                r = min(int(r * 1.5), 255)
                g = min(int(g * 1.5), 255)
                b = min(int(b * 1.5), 255)

            # Darken bright content
            elif brightness > bright_threshold:
                # Decrease brightness (scale factor can be adjusted)
                r = max(int(r * 0.5), 0)
                g = max(int(g * 0.5), 0)
                b = max(int(b * 0.5), 0)

            # Update the pixel
            pixels[x, y] = (r, g, b)

    # Save the modified image
    img.save(output_path)
    print(f"Processed image saved as '{output_path}'")

# Example usage
input_image_path = r'D:\Data\cv\shu_undergrad_admission_list.png'  # Path to the input image file
output_image_path = 'output_image.jpg'  # Path for the output image file
adjust_brightness(input_image_path, output_image_path)
