import os
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.restoration import estimate_sigma
import time
import csv


def calculate_noise_score(file_path):
    """
    Uses a function called estimate_sigma from the skimage library,
    which analyzes the image to determine the amount of noise present in each color channel.

    Noise in this context refers to the random variations in color or brightness in the image,
    which can make the image look grainy or speckled.
    """
    start_time = time.time()

    # Read the image
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or unable to read image")

    # Convert image to float
    image_float = img_as_float(image)

    # Estimate the noise standard deviation for each channel
    sigma_est_r = estimate_sigma(image_float[:, :, 0], channel_axis=None)
    sigma_est_g = estimate_sigma(image_float[:, :, 1], channel_axis=None)
    sigma_est_b = estimate_sigma(image_float[:, :, 2], channel_axis=None)

    # Average the sigma values to get a single overall noise score
    sigma_est = np.mean([sigma_est_r, sigma_est_g, sigma_est_b])
    print(f"Noise Score (Sigma): {sigma_est}")

    # Convert noise level to MOS scale (1-5)
    mos_score = noise_to_mos(sigma_est)

    end_time = time.time()
    computation_time = end_time - start_time

    return mos_score, f"{computation_time:.4f} s"


def noise_to_mos(sigma):
    """
    Converts noise standard deviation to a MOS score (1.0 - 5.0) with float precision.
    """
    if sigma < 0.02:
        mos_score = 5.0 - (sigma / 0.02) * 0.5
    elif sigma < 0.05:
        mos_score = 4.5 - ((sigma - 0.02) / 0.03) * 0.5
    elif sigma < 0.1:
        mos_score = 4.0 - ((sigma - 0.05) / 0.05) * 1.0
    elif sigma < 0.2:
        mos_score = 3.0 - ((sigma - 0.1) / 0.1) * 1.0
    else:
        mos_score = 2.0 - ((sigma - 0.2) / 0.3) * 1.0
        if mos_score < 1.0:
            mos_score = 1.0

    return mos_score


def process_folder(folder_path, output_csv_path):
    """
    Processes all images in a specified folder, calculates the noise score for each,
    and saves the results to a CSV file.
    """
    results = []
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_images = len(image_files)

    # Iterate over all files in the folder
    for idx, file_name in enumerate(image_files, 1):
        file_path = os.path.join(folder_path, file_name)
        try:
            mos_score, computation_time = calculate_noise_score(file_path)
            results.append([file_name, mos_score, computation_time])
            print(f"Processing image {idx}/{total_images}: {file_name}")
        except ValueError as e:
            print(f"Skipping {file_name}: {e}")

    # Save results to a CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "MOS Score", "Computation Time"])
        writer.writerows(results)


if __name__ == "__main__":
    folder_path = '../../VGG16/data/512x384'
    output_csv_path = 'Koniq10k_noise_scores.csv'
    process_folder(folder_path, output_csv_path)
    print(f"Results saved to {output_csv_path}")
