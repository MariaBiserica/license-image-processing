import os
import time

import cv2
import csv
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.svm import SVR
from joblib import load
from sklearn.impute import SimpleImputer
from repo.assessment_features.utils.scale_scores import scale_scores_in_csv


def convert_to_hsv(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def extract_color_moments(hsv_image):
    # Extract color moments for each channel
    moments = []
    for i in range(3):
        channel = hsv_image[:, :, i]
        moments.append(np.mean(channel))
        moments.append(np.std(channel))
        moments.append(skew(channel.flatten()))
    return moments


def log_gabor_filter(shape, wavelength, orientation, bandwidth=1):
    rows, cols = shape
    radius = np.sqrt((np.arange(-cols // 2, cols // 2) ** 2).reshape(1, -1) +
                     (np.arange(-rows // 2, rows // 2) ** 2).reshape(-1, 1))
    radius = np.fft.fftshift(radius)
    radius[0, 0] = 1  # Avoid division by zero

    theta = np.arctan2(-np.arange(-rows // 2, rows // 2).reshape(-1, 1),
                       np.arange(-cols // 2, cols // 2).reshape(1, -1))
    theta = np.fft.fftshift(theta)  # Shift zero frequency to center

    # Log-Gabor filter
    log_gabor = np.zeros(radius.shape, dtype=np.float32)
    log_gabor_filter_mask = radius > 0  # Apply filter only where radius is greater than 0
    log_gabor[log_gabor_filter_mask] = np.exp(-((np.log(radius[log_gabor_filter_mask] / wavelength)) ** 2) / (2 * np.log(bandwidth) ** 2))
    log_gabor[log_gabor_filter_mask] *= np.exp((np.cos(theta[log_gabor_filter_mask] - orientation) ** 2) / (-2 * np.log(bandwidth) ** 2))

    return log_gabor


def apply_log_gabor_filter(hsv_image):
    # Apply the log-Gabor filter to the V channel of HSV image
    v_channel = hsv_image[:, :, 2]
    v_channel = v_channel.astype(np.float32) / 255  # Normalize

    # Fourier transform of the V channel
    v_fft = np.fft.fft2(v_channel)

    # Define log-Gabor parameters and apply filter
    wavelength = 10  # Example wavelength
    orientation = 0  # Example orientation
    bandwidth = 1  # Example bandwidth
    log_gabor = log_gabor_filter(v_channel.shape, wavelength, orientation, bandwidth)
    filtered_fft = v_fft * log_gabor

    # Inverse Fourier transform to obtain the filtered V channel
    filtered_v = np.fft.ifft2(filtered_fft).real

    # This is a simplified implementation; in practice, you would generate multiple layers
    return [filtered_v]  # Placeholder for layers


def compute_mscn_coefficients(image_layer):
    """
    Compute Mean Subtracted Contrast Normalized (MSCN) coefficients for an image layer.
    """
    # Define a kernel for local mean calculation
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(image_layer, -1, kernel)

    # Compute the local variance
    local_var = cv2.filter2D(image_layer ** 2, -1, kernel) - local_mean ** 2

    # Ensure variance is non-zero
    local_var[local_var < 1e-10] = 1e-10

    # Compute MSCN coefficients
    mscn_coefficients = (image_layer - local_mean) / np.sqrt(local_var)

    return mscn_coefficients


def extract_texture_features_from_mscn(mscn_coefficients):
    """
    Extract texture features from MSCN coefficients.
    """
    # Calculate statistical features
    features = []
    features.append(np.mean(mscn_coefficients))
    features.append(np.var(mscn_coefficients))
    features.append(skew(mscn_coefficients.flatten()))
    features.append(kurtosis(mscn_coefficients.flatten()))

    return features


def extract_texture_features(image_layers):
    """
    Extract texture features from the MSCN coefficients of log-Gabor filtered layers.
    """
    all_texture_features = []
    for layer in image_layers:
        mscn_coefficients = compute_mscn_coefficients(layer)
        texture_features = extract_texture_features_from_mscn(mscn_coefficients)
        all_texture_features.extend(texture_features)

    return all_texture_features


def predict_quality(features, svr_model):
    # Use SVR to predict the quality score based on extracted features
    # n_samples = 1 -> one image
    # Assuming 'features' is already a 2D array with shape (1, n_features) after imputation
    return svr_model.predict(features)


# Main algorithm implementation
def calculate_chromatic_score(image_path, svr_model_path):
    hsv_image = convert_to_hsv(image_path)
    color_features = extract_color_moments(hsv_image)
    image_layers = apply_log_gabor_filter(hsv_image)
    texture_features = extract_texture_features(image_layers)
    features = np.concatenate((color_features, texture_features)).reshape(1, -1)

    print("Features array shape before imputation:", features.shape)
    print("Features array contents before imputation:", features)

    # Handle potential NaN values in features
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    features_imputed = imputer.fit_transform(features)  # Impute NaN values

    expected_feature_length = 9
    actual_feature_length = features_imputed.shape[1]

    # If there are missing features, add them as zeros or another constant value
    if actual_feature_length < expected_feature_length:
        padding = np.zeros((features_imputed.shape[0], expected_feature_length - actual_feature_length))
        features_imputed = np.hstack((features_imputed, padding))

    print("Features array shape after imputation:", features_imputed.shape)
    print("Features array contents after imputation:", features_imputed)

    svr_model = load(svr_model_path)  # Load the trained SVR model
    chromatic_quality_score = predict_quality(features_imputed, svr_model)
    return chromatic_quality_score[0]


def calculate_scaled_chromatic_score(image_path, csv_path, svr_model_path):
    """
    Calculate the chromatic score for a single image and scale it using the min and max from the CSV file.

    :param image_path: Path to the image file.
    :param csv_path: Path to the CSV file with contrast scores.
    :return: Scaled contrast score for the image.
    """
    start_time = time.time()  # Start timer

    # Calculate the contrast scores for the image
    overall_chromatic = calculate_chromatic_score(image_path, svr_model_path)

    # Load the CSV to find min and max scores for scaling
    df = pd.read_csv(csv_path)
    min_score, max_score = df['Chromatic_Score'].min(), df['Chromatic_Score'].max()

    # Define the new range for scaling
    new_min, new_max = 1, 5

    # Scale the overall contrast score
    scaled_chromatic_score = new_min + (new_max - new_min) * (overall_chromatic - min_score) / (max_score - min_score)
    print(f"Scaled Image Chromatic Score: {scaled_chromatic_score}")

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute duration

    return scaled_chromatic_score, f"{elapsed_time:.4f} s"  # Return score and time taken


def gather_scores_on_dataset(image_folder_path, output_csv_path, svr_model_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'Chromatic_Score'])

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(image_folder_path, filename)
                try:
                    overall_chromatic_score = calculate_chromatic_score(image_path, svr_model_path)
                    writer.writerow([filename, overall_chromatic_score])
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    # image_folder_path = '../../VGG16/data/512x384'
    image_folder_path = '../../alternate_VGG16/data/LIVE2/databaserelease2/LIVE_all'
    output_csv_path = 'LIVE2_chromatic_scores.csv'
    svr_model_path = 'svr_model.joblib'
    gather_scores_on_dataset(image_folder_path, output_csv_path, svr_model_path)

    scaled_csv_path = 'Scaled_LIVE2_chromatic_scores.csv'
    scale_scores_in_csv(output_csv_path, scaled_csv_path)

    # image_path = '../../VGG16/data/512x384/826373.jpg'
    # chromatic_score = calculate_chromatic_score(image_path, svr_model_path)
    # print(f"Chromatic Score: {chromatic_score}")
