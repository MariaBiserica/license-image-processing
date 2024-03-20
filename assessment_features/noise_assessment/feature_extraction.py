import numpy as np
import cv2
from scipy.stats import entropy
from scipy.fftpack import fft2
from scipy import signal


def calculate_variance(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate variance
    variance = np.var(gray_image)
    return variance


def calculate_power_spectral_entropy(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Fourier Transform
    f_transform = fft2(gray_image)
    # Compute power spectrum
    power_spectrum = np.abs(f_transform) ** 2
    # Normalize power spectrum
    ps_normalized = power_spectrum / np.sum(power_spectrum)
    # Calculate entropy
    ps_entropy = entropy(ps_normalized.flatten())
    return ps_entropy


def calculate_wavelet_std_dev(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use the Discrete Wavelet Transform from scipy
    # For simplicity, we'll apply a Haar wavelet-like operation using the convolve2d function.
    # This is more manual and less elegant than pywt but works as a basic alternative.
    haar_wavelet = np.array([[1, 1], [1, -1]])
    # Applying convolution to simulate wavelet transform
    c_2d = signal.convolve2d(gray_image, haar_wavelet, mode='same')
    # Calculate standard deviation of the transformed coefficients
    std_dev = np.std(c_2d)
    return std_dev
