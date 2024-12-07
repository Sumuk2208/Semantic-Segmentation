import cv2
import os

def apply_bilateral_filter_to_directory(input_directory, output_directory, d=9, sigma_color=75, sigma_space=75):
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through the image files and apply the filter
    for image_file in image_files:
        # Construct the full file path
        image_path = os.path.join(input_directory, image_file)

        # Read the image
        image = cv2.imread(image_path)

        # Apply the bilateral filter
        filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        # Save the filtered image to the output directory
        output_path = os.path.join(output_directory, f'filtered_{image_file}')
        cv2.imwrite(output_path, filtered_image)

        print(f"Filtered image saved to {output_path}")

# Example usage
input_directory = r'C:\Users\sumuk\PycharmProjects\CSCI635_project\results_camvid' # Replace with your input directory path
output_directory = r'C:\Users\sumuk\PycharmProjects\CSCI635_project\saved_masks'  # Replace with your desired output directory
apply_bilateral_filter_to_directory(input_directory, output_directory)
