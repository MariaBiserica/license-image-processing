import cv2
import numpy as np
import csv
import os
import time
import pandas as pd


def jnd_model(l_channel):
    """
    Implement the Just Noticeable Difference (JND) model for the luminance channel.
    This function should calculate the JND value for each pixel based on the paper's formula.
    """
    # Constants from the JND model
    T0 = 17
    gamma = 3 / 128
    lambda_ = 0.5

    # Ensure l_channel is float for correct division
    l_channel_float = l_channel.astype(np.float32)

    # Calculate JND for each pixel based on background luminance
    jnd = np.zeros_like(l_channel_float)
    jnd[l_channel_float <= 127] = T0 * (1 - (l_channel_float[l_channel_float <= 127] / 127.0) ** lambda_) + 3
    jnd[l_channel_float > 127] = gamma * (l_channel_float[l_channel_float > 127] - 127) + 3

    return jnd


def calculate_sad(jnd_values, window_size=3):
    """
    Calculate Sum of Absolute Differences (SAD) in a local region.
    This is a simplified version and does not directly apply windowing.
    For accurate implementation, consider each local region specifically.
    """
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2)
    sad = cv2.filter2D(jnd_values, -1, kernel)
    return np.abs(sad - np.mean(jnd_values))


def calculate_fjnd(jnd_values, threshold=3):
    """
    Calculate the factor that satisfies the JND visibility threshold.
    This is a simplified version for demonstration purposes.
    """
    max_jnd = np.max(jnd_values)
    min_jnd = np.min(jnd_values)
    return (max_jnd - min_jnd) / threshold


# Calculate the local contrast factor (C) for the luminance channel
def calculate_luminance_contrast(L_channel, window_size=3):
    """
    Calculate the luminance contrast of an image based on the JND model.
    """
    jnd_values = jnd_model(L_channel)
    sad_values = calculate_sad(jnd_values, window_size=window_size)
    fjnd_values = calculate_fjnd(jnd_values)
    contrast = np.mean(sad_values * fjnd_values)
    return contrast


def calculate_rrf(I, psi, eta=1, voff=0.1):
    """
    Calculate the Region Response Factor (RRF) based on brightness and luminance.

    :param I: Luminance of the region.
    :param psi: Brightness value of the region.
    :param eta: Weight factor.
    :param voff: Offset value to prevent the contrast from being estimated as 0.
    :return: The RRF value for the region.
    """
    # Ensure the division by zero is prevented
    psi = np.maximum(psi, 0.1)
    rrf = np.where(I >= psi, ((I - psi) / psi) ** eta + voff,
                   ((psi - I) / psi) ** eta + voff)

    return rrf


def calculate_chromatic_contrast(channel, L_channel, window_size=3, eta=1, voff=0.1):
    # Convert the input channel to float for precise computation
    channel = channel.astype(np.float32)

    # Define a local window for computing local brightness
    local_window = window_size * 2
    kernel = np.ones((local_window, local_window), np.float32) / (local_window ** 2)

    # Compute local brightness (ψ) as the mean luminance in the local region
    local_brightness = cv2.filter2D(L_channel, -1, kernel)

    # Calculate RRF based on local_brightness and global luminance
    rrf = calculate_rrf(L_channel, local_brightness, eta, voff)

    # Calculate the local contrast of the channel
    channel_local_contrast = cv2.filter2D(channel, -1, kernel) - channel

    # Apply RRF to the local contrast of the channel
    weighted_channel_contrast = rrf * np.abs(channel_local_contrast)

    # Compute the chromatic contrast as the mean of the weighted channel contrast
    chromatic_contrast = np.mean(weighted_channel_contrast)

    return chromatic_contrast


def calculate_blue_yellow_contrast(B_channel, L_channel):
    return calculate_chromatic_contrast(B_channel, L_channel)


def calculate_red_green_contrast(A_channel, L_channel):
    return calculate_chromatic_contrast(A_channel, L_channel)


def calculate_contrast(image_path):
    """
    Calculate both luminance and color contrast of an image.
    """
    # Load the image in BGR color space
    image = cv2.imread(image_path)

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split LAB image into L, A, and B channels
    L_channel, A_channel, B_channel = cv2.split(lab_image)

    # Calculate luminance contrast
    luminance_contrast = calculate_luminance_contrast(L_channel.astype(np.float32))

    # Calculate Ccb and Ccr
    blue_yellow_contrast = calculate_blue_yellow_contrast(B_channel, L_channel)
    red_green_contrast = calculate_red_green_contrast(A_channel, L_channel)

    return luminance_contrast, blue_yellow_contrast, red_green_contrast


def calculate_overall_contrast(luminance_contrast, blue_yellow_contrast, red_green_contrast, alpha=0.4, beta=0.3,
                               theta=0.3):
    """
    Calculate the overall image contrast considering the luminance and chromatic components.

    :param luminance_contrast: Luminance contrast value.
    :param blue_yellow_contrast: Chromatic contrast in the blue-yellow domain.
    :param red_green_contrast: Chromatic contrast in the red-green domain.
    :param alpha: Weight for luminance contrast.
    :param beta: Weight for blue-yellow chromatic contrast.
    :param theta: Weight for red-green chromatic contrast.
    :return: Overall image contrast.
    """
    # Ensure weights sum to 1
    alpha += beta + theta
    beta /= alpha
    theta /= alpha
    alpha = 1 - (beta + theta)

    # Calculate the overall contrast
    overall_contrast = (luminance_contrast ** alpha) * (blue_yellow_contrast ** beta) * (red_green_contrast ** theta)

    return overall_contrast


def calculate_contrast_score(image_path):
    luminance_contrast, blue_yellow_contrast, red_green_contrast = calculate_contrast(image_path)
    print(f"CL - Luminance Contrast: {luminance_contrast}\n"
          f"Ccb - Blue Yellow Contrast: {blue_yellow_contrast}\n"
          f"Ccr - Red Green Contrast: {red_green_contrast}")

    overall_contrast = calculate_overall_contrast(luminance_contrast, blue_yellow_contrast, red_green_contrast)
    print(f"Cimage - Overall Image Contrast Score: {overall_contrast}")

    return overall_contrast


def calculate_scaled_contrast_score(image_path, csv_path):
    """
    Calculate the contrast score for a single image and scale it using the min and max from the CSV file.

    :param image_path: Path to the image file.
    :param csv_path: Path to the CSV file with contrast scores.
    :return: Scaled contrast score for the image.
    """
    start_time = time.time()  # Start timer

    # Calculate the contrast scores for the image
    overall_contrast = calculate_contrast_score(image_path)

    # Load the CSV to find min and max scores for scaling
    df = pd.read_csv(csv_path)
    min_score, max_score = df['Overall_Contrast_Score'].min(), df['Overall_Contrast_Score'].max()

    # Define the new range for scaling
    new_min, new_max = 1, 5

    # Scale the overall contrast score
    scaled_contrast_score = new_min + (new_max - new_min) * (overall_contrast - min_score) / (max_score - min_score)
    print(f"Contrast Max: {max_score}")
    print(f"Contrast Min: {min_score}")
    print(f"Scaled Image Contrast Score: {scaled_contrast_score}")

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute duration

    return scaled_contrast_score, f"{elapsed_time:.4f} s"  # Return score and time taken


def gather_scores_on_dataset(image_folder_path, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'Overall_Contrast_Score'])

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(image_folder_path, filename)
                try:
                    luminance_contrast, blue_yellow_contrast, red_green_contrast = calculate_contrast(image_path)
                    overall_contrast_score = calculate_overall_contrast(luminance_contrast, blue_yellow_contrast, red_green_contrast)
                    writer.writerow([filename, overall_contrast_score])
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


def main():
    # Update this path to the correct folder where your images are stored
    # image_folder_path = '../../alternate_VGG16/data/LIVE2/databaserelease2/LIVE_all'

    # Update this path to where you want to save the CSV file
    # output_csv_path = 'LIVE2_contrast_scores.csv'

    # gather_scores_on_dataset(image_folder_path, output_csv_path)

    # Optionally call the scaling function here if you want it to be part of the main process
    # Update this path to where you want to save the scaled scores CSV file
    # scaled_csv_path = 'Scaled_LIVE2_contrast_scores.csv'
    # scale_scores_in_csv(output_csv_path, scaled_csv_path)

    image_path = '../../alternate_VGG16/data/Koniq_10k/512x384/826373.jpg'
    scores_csv_path = 'Koniq10k_contrast_scores.csv'
    calculate_scaled_contrast_score(image_path, scores_csv_path)


if __name__ == "__main__":
    main()
