import os
import cv2
import csv
import numpy as np
import pandas as pd
from joblib import dump, load
from repo.assessment_features.noise_assessment.feature_extraction import (
    calculate_variance, calculate_power_spectral_entropy, calculate_wavelet_std_dev)
from repo.assessment_features.noise_assessment.extreme_learning_machine import ELM
from repo.assessment_features.utils.scale_scores import scale_scores_in_csv


def extract_features_for_dataset(df, images_dir):
    x = []
    y = []
    print("Starting feature extraction for dataset...")
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"Processing image {index + 1}/{len(df)}")
        image_path = os.path.join(images_dir, row['image_name'])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image at {image_path}. Skipping.")
            continue

        variance = calculate_variance(image)
        ps_entropy = calculate_power_spectral_entropy(image)
        wavelet_std_dev = calculate_wavelet_std_dev(image)

        x.append([variance, ps_entropy, wavelet_std_dev])
        y.append(row['MOS'])
    print("Feature extraction completed.")
    return np.array(x), np.array(y)


def train_elm_model(x_train, y_train, save_path='elm_model.joblib'):
    print("Initializing and training the ELM model...")
    elm_model = ELM(n_hidden_units=100, alpha=0.1)
    elm_model.fit(x_train, y_train)
    print("ELM model training completed.")

    # Save the trained model to disk
    dump(elm_model, save_path)
    print(f"Model saved to {save_path}.")

    return elm_model


def load_elm_model(load_path='elm_model.joblib'):
    print(f"Loading model from {load_path}...")
    elm_model = load(load_path)
    return elm_model


def calculate_noise_score(elm_model, image):
    print("Predicting quality score based on noise for the new image...")
    variance = calculate_variance(image)
    ps_entropy = calculate_power_spectral_entropy(image)
    wavelet_std_dev = calculate_wavelet_std_dev(image)

    features = np.array([variance, ps_entropy, wavelet_std_dev]).reshape(1, -1)
    quality_score = elm_model.predict(features)
    return quality_score


def calculate_scaled_noise_score(elm_model, image, csv_path):
    # Calculate the contrast scores for the image
    overall_noise = calculate_noise_score(elm_model, image)
    print(f"Image Noise Score: {overall_noise[0]}")

    # Load the CSV to find min and max scores for scaling
    df = pd.read_csv(csv_path)
    min_score, max_score = df['Noise_Score'].min(), df['Noise_Score'].max()

    # Define the new range for scaling
    new_min, new_max = 1, 5

    # Scale the overall contrast score
    scaled_score = new_min + (new_max - new_min) * (overall_noise[0] - min_score) / (max_score - min_score)
    print(f"Scaled Image Noise Score: {scaled_score}")

    return scaled_score


def gather_scores_on_dataset(image_folder_path, output_csv_path, model):
    """
    Process images in a folder, calculate scores, and save them to a CSV.
    """
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'Noise_Score'])

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(image_folder_path, filename)
                image = cv2.imread(image_path)
                noise_score = calculate_noise_score(model, image)
                writer.writerow([filename, noise_score[0]])
    print("Scores gathered and saved to:", output_csv_path)


def main():
    dataset_dir = '../../VGG16/data/512x384'
    model_path = 'elm_model.joblib'
    if os.path.exists(model_path):
        elm_model = load_elm_model(model_path)
    else:
        dataset_csv_path = '../../VGG16/data/koniq10k_scores_and_distributions.csv'

        print("Loading dataset...")
        df = pd.read_csv(dataset_csv_path)
        print(f"Dataset loaded. Total images: {len(df)}")

        x_train, y_train = extract_features_for_dataset(df, dataset_dir)
        elm_model = train_elm_model(x_train, y_train, save_path=model_path)

    # output_csv_path = 'Koniq10k_noise_scores.csv'
    # scaled_output_csv_path = 'Scaled_Koniq10k_noise_scores.csv'
    # gather_scores_on_dataset(dataset_dir, output_csv_path, elm_model)
    # scale_scores_in_csv(output_csv_path, scaled_output_csv_path)

    image_path = '../../VGG16/data/512x384/826373.jpg'
    scores_csv_path = 'Koniq10k_noise_scores.csv'
    image = cv2.imread(image_path)
    calculate_scaled_noise_score(elm_model, image, scores_csv_path)


if __name__ == "__main__":
    main()
