import cv2
import numpy as np
import pandas as pd
import os
from repo.brisque_release_online.brisque_master.brisque.brisque_algorithm import BRISQUE


def get_ground_truth_scores(csv_path):
    """
    Extract ground truth scores from a CSV file for the Koniq-10k database.

    :param csv_path: The path to the CSV file with image scores.
    :return: A dictionary with image filenames as keys and ground truth scores as values.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Assuming the CSV has columns 'image_name' and 'score'
    # Replace 'image_name' and 'score' with the actual column names
    ground_truth_scores = df.set_index('image_name')['MOS'].to_dict()

    return ground_truth_scores


def measure_brisque(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Check if the image was loaded correctly
    assert img is not None, "Error loading image. Check the path."

    # Create a BRISQUE object
    obj = BRISQUE(url=False)

    # Calculate the BRISQUE score
    quality_score = obj.score(img)

    return quality_score


def calculate_scaled_brisque_score(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Check if the image was loaded correctly
    assert img is not None, "Error loading image. Check the path."

    # Create a BRISQUE object
    obj = BRISQUE(url=False)

    # Calculate the BRISQUE score
    quality_score = obj.score(img)

    # Scale the BRISQUE (100 - 0) score to MOS (1 - 5)
    if not (0 <= quality_score <= 100):
        raise ValueError("BRISQUE score should be between 0 and 100.")

    # Linearly scale the BRISQUE score to MOS
    # BRISQUE 0 (best) -> MOS 5 (excellent)
    # BRISQUE 100 (worst) -> MOS 1 (bad)
    scaled_quality_score = 5 - 4 * (quality_score / 100)

    return scaled_quality_score


def brisque_to_mos(brisque_score):
    """
    Scale BRISQUE score to MOS.

    :param brisque_score: The BRISQUE score to convert.
    :return: A MOS score ranging from 1 (bad) to 5 (excellent).
    """
    # Ensure BRISQUE score is within the expected range
    if not (0 <= brisque_score <= 100):
        raise ValueError("BRISQUE score should be between 0 and 100.")

    # Linearly scale the BRISQUE score to MOS
    # BRISQUE 0 (best) -> MOS 5 (excellent)
    # BRISQUE 100 (worst) -> MOS 1 (bad)
    mos_score = 5 - 4 * (brisque_score / 100)
    return mos_score


def evaluate_performance(brisque_instance, image_path, ground_truth_scores_dict):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    predicted_score = brisque_instance.score(img)

    image_name = os.path.basename(image_path)
    ground_truth_score = ground_truth_scores_dict.get(image_name)
    if ground_truth_score is None:
        raise ValueError(f"No ground truth score found for image {image_name}")

    # Use the BRISQUE instance to evaluate the performance
    # performance_metrics = brisque_instance.test_performance_metrics(np.array([predicted_score]),
    #                                                                 np.array([ground_truth_score]))

    # return performance_metrics


def main():
    csv_path = "..\\..\\..\\VGG16\\data\\koniq10k_scores_and_distributions.csv"
    ground_truth_scores_dict = get_ground_truth_scores(csv_path)

    image_path = "..\\..\\..\\VGG16\\data\\512x384\\826373.jpg"
    image_name = os.path.basename(image_path)  # Extract the filename

    # Get the ground truth score for the given image
    ground_truth_score = ground_truth_scores_dict.get(image_name)
    if ground_truth_score is None:
        print(f"No ground truth score found for image {image_name}")
        return

    brisque_obj = BRISQUE(url=False)
    quality_score = brisque_obj.score(cv2.imread(image_path, cv2.IMREAD_COLOR))
    scaled_brisque_score = brisque_to_mos(quality_score)
    print(f'BRISQUE Quality Score: {quality_score:.4f}')
    print(f'BRISQUE Scaled Quality Score: {scaled_brisque_score:.4f}')
    print(f'Ground Truth Score: {ground_truth_score:.4f}')

    # Evaluate the performance for a single image path
    # performance_results = evaluate_performance(brisque_obj, image_path, ground_truth_scores_dict)
    # print("Performance Metrics:")
    # for metric, value in performance_results.items():
    #     print(f'{metric}: {value:.4f}')


if __name__ == "__main__":
    main()
