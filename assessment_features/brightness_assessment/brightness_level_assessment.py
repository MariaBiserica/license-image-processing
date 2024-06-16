import time
import cv2
import csv
import os
import numpy as np
import pandas as pd
from repo.assessment_features.utils_custom.scale_scores import scale_scores_in_csv


def convert_to_linear_light(image):
    """
    Assuming image is in sRGB domain, convert to linear light (RGB).
    This function needs to be adapted if the image is in a different format.
    """
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def compute_rgb_max(image):
    """
    Compute the RGBmax matrix, keeping the maximum value among the R, G, and B channels for each pixel.
    """
    return np.max(image, axis=2)


def apply_spatial_weighting(rgb_max):
    """
    Apply a spatial weighting based on pixel proximity to the image's center.
    This implementation uses a Gaussian kernel for simplicity.
    """
    rows, cols = rgb_max.shape
    x = np.linspace(-cols//2, cols//2, cols)
    y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(x, y)
    d = np.sqrt(X**2 + Y**2)
    sigma, mu = cols/4, 0.0  # Adjust sigma for the desired weighting effect
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    weighted_rgb_max = rgb_max * g
    return weighted_rgb_max


def calculate_rms(weighted_rgb_max):
    """
    Calculate the Root Mean Square (RMS) of the weighted RGBmax matrix.
    """
    rms = np.sqrt(np.mean(np.square(weighted_rgb_max)))
    return rms


def calculate_brightness_score(image_path):
    """
    Quantify the brightness of an image based on the proposed HDR image brightness quantification method.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize and convert to RGB
    linear_image = convert_to_linear_light(image)
    rgb_max = compute_rgb_max(linear_image)
    weighted_rgb_max = apply_spatial_weighting(rgb_max)
    brightness_score = calculate_rms(weighted_rgb_max)
    return brightness_score


def calculate_scaled_brightness_score(image_path, csv_path):
    """
    Calculate the contrast score for a single image and scale it using the min and max from the CSV file.

    :param image_path: Path to the image file.
    :param csv_path: Path to the CSV file with contrast scores.
    :return: Scaled contrast score for the image.
    """
    start_time = time.time()  # Start timer

    # Calculate the contrast scores for the image
    overall_brightness = calculate_brightness_score(image_path)

    # Load the CSV to find min and max scores for scaling
    df = pd.read_csv(csv_path)
    min_score, max_score = df['Brightness_Score'].min(), df['Brightness_Score'].max()

    # Define the new range for scaling
    new_min, new_max = 1, 5

    # Scale the overall contrast score
    scaled_brightness_score = new_min + (new_max - new_min) * (overall_brightness - min_score) / (max_score - min_score)
    print(f"Brightness Max: {max_score}")
    print(f"Brightness Min: {min_score}")
    print(f"Scaled Image Brightness Score: {scaled_brightness_score}")

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute duration

    return scaled_brightness_score, f"{elapsed_time:.4f} s"  # Return score and time taken


def gather_scores_on_dataset(image_folder_path, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'Brightness_Score'])

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(image_folder_path, filename)
                try:
                    overall_brightness_score = calculate_brightness_score(image_path)
                    writer.writerow([filename, overall_brightness_score])
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


def main():
    # image_folder_path = '../../VGG16/data/512x384'
    image_folder_path = '../../alternate_VGG16/data/LIVE2/databaserelease2/LIVE_all'
    output_csv_path = 'LIVE2_brightness_scores.csv'
    gather_scores_on_dataset(image_folder_path, output_csv_path)

    scaled_csv_path = 'Scaled_LIVE2_brightness_scores.csv'
    scale_scores_in_csv(output_csv_path, scaled_csv_path)

    # image_path = '../../VGG16/data/512x384/826373.jpg'
    # brightness_score = calculate_brightness_score(image_path)
    # print(f"Brightness score: {brightness_score}")


if __name__ == "__main__":
    main()
