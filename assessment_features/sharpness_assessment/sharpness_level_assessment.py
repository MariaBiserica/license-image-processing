import time
import cv2
import csv
import os
import numpy as np
import pandas as pd
from repo.assessment_features.utils_custom.scale_scores import scale_scores_in_csv


def calculate_sharpness_score(image_path):
    """
    Calculate the Laplacian variance (LAPE metric - Laplacian Energy) of an image,
    which is an indicator of its sharpness. A higher variance indicates a sharper image.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image was loaded
    if image is None:
        raise ValueError("Image not loaded properly. Check the file path.")

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Calculate the variance
    variance = laplacian.var()

    return variance


def calculate_scaled_sharpness_score(image_path, csv_path):
    """
    Calculate the contrast score for a single image and scale it using the min and max from the CSV file,
    focusing on the range where most values lie.

    :param image_path: Path to the image file.
    :param csv_path: Path to the CSV file with contrast scores.
    :return: Scaled contrast score for the image.
    """
    start_time = time.time()  # Start timer

    # Calculate the contrast scores for the image
    overall_sharpness = calculate_sharpness_score(image_path)

    # Load the CSV to find min and max scores for scaling
    df = pd.read_csv(csv_path)

    # Calculate the 5th and 95th percentiles to exclude outliers
    min_score = np.percentile(df['Sharpness_Score'], 5)
    max_score = np.percentile(df['Sharpness_Score'], 95)

    # Define the new range for scaling
    new_min, new_max = 1, 5

    # Scale the overall contrast score
    scaled_sharpness_score = new_min + (new_max - new_min) * (overall_sharpness - min_score) / (max_score - min_score)
    scaled_sharpness_score = max(new_min, min(scaled_sharpness_score, new_max))  # Ensure the score is within the range

    print(f"Sharpness Max (95th percentile): {max_score}")
    print(f"Sharpness Min (5th percentile): {min_score}")
    print(f"Image Sharpness Score: {overall_sharpness}")
    print(f"Scaled Image Sharpness Score: {scaled_sharpness_score}")

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute duration

    return scaled_sharpness_score, f"{elapsed_time:.4f} s"  # Return score and time taken


def gather_scores_on_dataset(image_folder_path, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'Sharpness_Score'])

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(image_folder_path, filename)
                try:
                    overall_sharpness_score = calculate_sharpness_score(image_path)
                    writer.writerow([filename, overall_sharpness_score])
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


def main():
    # image_folder_path = '../../VGG16/data/512x384'
    image_folder_path = '../../alternate_VGG16/data/LIVE2/databaserelease2/LIVE_all'
    output_csv_path = 'LIVE2_sharpness_scores.csv'
    gather_scores_on_dataset(image_folder_path, output_csv_path)

    scaled_csv_path = 'Scaled_LIVE2_sharpness_scores.csv'
    scale_scores_in_csv(output_csv_path, scaled_csv_path)

    # image_path = '../../VGG16/data/512x384/826373.jpg'
    # sharpness_score = calculate_sharpness_score(image_path)
    # print(f"Laplacian variance (LAPE metric): {sharpness_score}")


if __name__ == "__main__":
    main()
