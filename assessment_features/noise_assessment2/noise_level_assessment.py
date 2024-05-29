import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.restoration import estimate_sigma
import time


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


# Example usage
if __name__ == "__main__":
    # file_path = '../../VGG16/data/512x384/826373.jpg'  # Original img
    file_path = r'C:\Users\maria\Downloads\modified_image_1717018351182.jpg'  # Denoised img
    mos_score, computation_time = calculate_noise_score(file_path)
    print(f"MOS Score: {mos_score:.4f}")  # Print MOS score with 4 decimal places
    print(f"Computation Time: {computation_time} seconds")
