import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from joblib import dump
from sklearn.impute import SimpleImputer

from repo.assessment_features.chromatic_assessment.chromatic_level_assessment import convert_to_hsv, \
    apply_log_gabor_filter, extract_color_moments, extract_texture_features


def load_dataset(scores_csv_path):
    """
    Load the quality scores dataset.
    """
    print("Loading dataset...")
    df = pd.read_csv(scores_csv_path)
    print(f"Dataset loaded with {len(df)} entries.")
    return df


def extract_features_for_dataset(image_dir, image_names):
    """
    Extract features for a list of image names, using the specified directory.
    """
    print("Starting feature extraction...")
    features = []
    for index, image_name in enumerate(image_names):
        print(f"Processing image {index + 1}/{len(image_names)}: {image_name}")
        image_path = os.path.join(image_dir, image_name)
        hsv_image = convert_to_hsv(image_path)
        color_features = extract_color_moments(hsv_image)
        image_layers = apply_log_gabor_filter(hsv_image)
        texture_features = extract_texture_features(image_layers)
        features.append(color_features + texture_features)

    print("Feature extraction completed.")
    return np.array(features)


def handle_missing_features(features):
    """Handle features that are completely missing."""
    # Identify features with all NaN values
    all_nan_features = np.isnan(features).all(axis=0)
    if all_nan_features.any():
        print(f"Features completely missing: {np.where(all_nan_features)[0]}")
        # Optional: Remove completely missing features
        features = features[:, ~all_nan_features]
    return features, ~all_nan_features


def train_svr_model(X_train, y_train):
    """
    Train an SVR model on the provided training data.
    """
    print("Training SVR model...")
    svr_model = SVR(C=1.0, epsilon=0.2)
    svr_model.fit(X_train, y_train)
    print("SVR model trained successfully.")
    return svr_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the SVR model on the test set.
    """
    print("Evaluating model...")
    predictions = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    srcc = spearmanr(y_test, predictions).correlation
    plcc = pearsonr(y_test, predictions)[0]
    mae = mean_absolute_error(y_test, predictions)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Spearman Rank Correlation Coefficient (SRCC): {srcc}")
    print(f"Pearson Linear Correlation Coefficient (PLCC): {plcc}")
    print(f"Mean Absolute Error (MAE): {mae}")


image_dir = '../../VGG16/data/512x384'
scores_csv_path = '../../VGG16/data/koniq10k_scores_and_distributions.csv'

print("Script started.")
df = load_dataset(scores_csv_path)
X = df['image_name']
y = df['MOS']

print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Extracting features for training and testing sets...")
X_train_features = extract_features_for_dataset(image_dir, X_train)
X_test_features = extract_features_for_dataset(image_dir, X_test)

# Handle completely missing features
X_train_features, valid_features_mask = handle_missing_features(X_train_features)
X_test_features = X_test_features[:, valid_features_mask]

print("Checking for NaN values in the training features...")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
if np.isnan(X_train_features).any():
    print("NaN values found in the training features. Handling NaN values...")
    # Choose a strategy to handle NaN values, such as imputation
    X_train_features = imputer.fit_transform(X_train_features)
    print("NaN values handled in training features.")

print("Checking for NaN values in the testing features...")
if np.isnan(X_test_features).any():
    print("NaN values found in the testing features. Handling NaN values...")
    X_test_features = imputer.transform(X_test_features)
    print("NaN values handled in testing features.")

print("Training the SVR model...")
svr_model = train_svr_model(X_train_features, y_train)

print("Saving the trained model...")
dump(svr_model, 'svr_model.joblib')

print("Evaluating the model...")
evaluate_model(svr_model, X_test_features, y_test)
print("Script finished.")
