import io
import os
import cv2
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify, json
from werkzeug.utils import secure_filename
from repo.assessment_features.noise_assessment.noise_level_assessment import (load_elm_model,
                                                                              calculate_scaled_noise_score)
from repo.assessment_features.contrast_assessment.contrast_level_assessment import calculate_scaled_contrast_score
from repo.assessment_features.brightness_assessment.brightness_level_assessment import calculate_scaled_brightness_score
from repo.assessment_features.sharpness_assessment.sharpness_level_assessment import calculate_scaled_sharpness_score
from repo.assessment_features.chromatic_assessment.chromatic_level_assessment import calculate_scaled_chromatic_score
from repo.brisque_release_online.brisque_master.brisque.brisque_quality import calculate_scaled_brisque_score
from repo.niqe_release_online.niqe import calculate_scaled_niqe_score
from repo.ilniqe_release_online.ilniqe_master.ilniqe import calculate_scaled_ilniqe_score
from repo.VGG16.vgg16_quality_score import measure_vgg16

from repo.modification_features.image_modification_spline_tool import modify_image

app = Flask(__name__)

# Directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Path to the CSV file with contrast scores for scaling
NOISE_CSV_PATH = '../assessment_features/noise_assessment/Koniq10k_noise_scores.csv'
CONTRAST_CSV_PATH = '../assessment_features/contrast_assessment/Koniq10k_contrast_scores.csv'
BRIGHTNESS_CSV_PATH = '../assessment_features/brightness_assessment/Koniq10k_brightness_scores.csv'
SHARPNESS_CSV_PATH = '../assessment_features/sharpness_assessment/Koniq10k_sharpness_scores.csv'
CHROMATIC_CSV_PATH = '../assessment_features/chromatic_assessment/Koniq10k_chromatic_scores.csv'
SVR_MODEL_PATH = '../assessment_features/chromatic_assessment/svr_model.joblib'
NIQE_SCORES_CSV_PATH = '../analysis/analyze_niqe/niqe_scores_Koniq10k.csv'
ILNIQE_SCORES_CSV_PATH = '../analysis/analyze_ilniqe/ilniqe_scores_Koniq10k.csv'

# Load your model (adjust 'elm_model.joblib' as needed)
elm_model = load_elm_model('../assessment_features/noise_assessment/elm_model.joblib')


@app.route('/predict', methods=['POST'])
def predict_quality():
    print("Received request for quality prediction")

    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    if 'metrics' not in request.form:
        return jsonify({'error': 'Missing metrics'}), 400

    # Process file
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Read selected metrics
    selected_metrics = set(json.loads(request.form['metrics']))

    # Load image
    image = cv2.imread(file_path)

    # Initialize results dictionary
    results = {}

    # Conditionally calculate scores
    if 'Noise' in selected_metrics:
        noise_score, noise_time = calculate_scaled_noise_score(elm_model, image, NOISE_CSV_PATH)
        results['noise_score'] = f"{noise_score:.4f}"
        results['noise_time'] = noise_time
    if 'Contrast' in selected_metrics:
        contrast_score, contrast_time = calculate_scaled_contrast_score(file_path, CONTRAST_CSV_PATH)
        results['contrast_score'] = f"{contrast_score:.4f}"
        results['contrast_time'] = contrast_time
    if 'Brightness' in selected_metrics:
        brightness_score, brightness_time = calculate_scaled_brightness_score(file_path, BRIGHTNESS_CSV_PATH)
        results['brightness_score'] = f"{brightness_score:.4f}"
        results['brightness_time'] = brightness_time
    if 'Sharpness' in selected_metrics:
        sharpness_score, sharpness_time = calculate_scaled_sharpness_score(file_path, SHARPNESS_CSV_PATH)
        results['sharpness_score'] = f"{sharpness_score:.4f}"
        results['sharpness_time'] = sharpness_time
    if 'Chromatic Quality' in selected_metrics:
        chromatic_score, chromatic_time = calculate_scaled_chromatic_score(file_path, CHROMATIC_CSV_PATH, SVR_MODEL_PATH)
        results['chromatic_score'] = f"{chromatic_score:.4f}"
        results['chromatic_time'] = chromatic_time
    if 'BRISQUE' in selected_metrics:
        brisque_score, brisque_time = calculate_scaled_brisque_score(file_path)
        results['brisque_score'] = f"{brisque_score:.4f}"
        results['brisque_time'] = brisque_time
    if 'NIQE' in selected_metrics:
        niqe_score, niqe_time = calculate_scaled_niqe_score(file_path, NIQE_SCORES_CSV_PATH)
        results['niqe_score'] = f"{niqe_score:.4f}"
        results['niqe_time'] = niqe_time
    if 'ILNIQE' in selected_metrics:
        ilniqe_score, ilniqe_time = calculate_scaled_ilniqe_score(file_path, ILNIQE_SCORES_CSV_PATH)
        results['ilniqe_score'] = f"{ilniqe_score:.4f}"
        results['ilniqe_time'] = ilniqe_time
    if 'VGG16' in selected_metrics:
        vgg16_score, vgg16_time = measure_vgg16(file_path)
        results['vgg16_score'] = f"{vgg16_score:.4f}"
        results['vgg16_time'] = vgg16_time

    # Clean up the uploaded image after processing
    os.remove(file_path)

    # Return the calculated scores
    return jsonify(results)


@app.route('/predict_batch', methods=['POST'])
def predict_quality_batch():
    print("Received request for batch quality prediction")

    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    if 'metric' not in request.form:
        return jsonify({'error': 'Missing metrics'}), 400

    metric = request.form['metric']
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    results = {}

    # Example for one metric
    if metric == 'Noise':
        noise_score, noise_time = calculate_scaled_noise_score(elm_model, image, NOISE_CSV_PATH)
        results['noise_score'] = f"{noise_score:.4f}"
        results['noise_time'] = noise_time
    if metric == 'Contrast':
        contrast_score, contrast_time = calculate_scaled_contrast_score(file_path, CONTRAST_CSV_PATH)
        results['contrast_score'] = f"{contrast_score:.4f}"
        results['contrast_time'] = contrast_time
    if metric == 'Brightness':
        brightness_score, brightness_time = calculate_scaled_brightness_score(file_path, BRIGHTNESS_CSV_PATH)
        results['brightness_score'] = f"{brightness_score:.4f}"
        results['brightness_time'] = brightness_time
    if metric == 'Sharpness':
        sharpness_score, sharpness_time = calculate_scaled_sharpness_score(file_path, SHARPNESS_CSV_PATH)
        results['sharpness_score'] = f"{sharpness_score:.4f}"
        results['sharpness_time'] = sharpness_time
    if metric == 'Chromatic Quality':
        chromatic_score, chromatic_time = calculate_scaled_chromatic_score(file_path, CHROMATIC_CSV_PATH, SVR_MODEL_PATH)
        results['chromatic_score'] = f"{chromatic_score:.4f}"
        results['chromatic_time'] = chromatic_time
    if metric == 'BRISQUE':
        brisque_score, brisque_time = calculate_scaled_brisque_score(file_path)
        results['brisque_score'] = f"{brisque_score:.4f}"
        results['brisque_time'] = brisque_time
    if metric == 'NIQE':
        niqe_score, niqe_time = calculate_scaled_niqe_score(file_path, NIQE_SCORES_CSV_PATH)
        results['niqe_score'] = f"{niqe_score:.4f}"
        results['niqe_time'] = niqe_time
    if metric == 'ILNIQE':
        ilniqe_score, ilniqe_time = calculate_scaled_ilniqe_score(file_path, ILNIQE_SCORES_CSV_PATH)
        results['ilniqe_score'] = f"{ilniqe_score:.4f}"
        results['ilniqe_time'] = ilniqe_time
    if metric == 'VGG16':
        vgg16_score, vgg16_time = measure_vgg16(file_path)
        results['vgg16_score'] = f"{vgg16_score:.4f}"
        results['vgg16_time'] = vgg16_time

    os.remove(file_path)  # Clean up the uploaded image after processing
    return jsonify(results)


@app.route('/modify-image-spline', methods=['POST'])
def modify_image_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    control_points_str = request.form['control_points']

    # Convert control points from string to list of tuples
    control_points = [tuple(map(int, point.split(','))) for point in control_points_str.split(';')]

    image = Image.open(image_file.stream)
    modified_image = modify_image(image, control_points)

    img_io = io.BytesIO()
    modified_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_gaussian_blur', methods=['POST'])
def apply_gaussian_blur():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    blur_amount = float(request.form.get('blur_amount', 15))
    blur_amount = round(blur_amount)  # Ensure blur amount is an integer

    # Make sure blur_amount is positive and odd
    if blur_amount < 1:
        blur_amount = 1
    if blur_amount % 2 == 0:
        blur_amount += 1

    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", blurred_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_edge_detection', methods=['POST'])
def apply_edge_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", edges)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_color_space_conversion', methods=['POST'])
def apply_color_space_conversion():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    color_space = request.form.get('color_space', 'HSV')
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if color_space == 'HSV':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == 'YCrCb':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        return jsonify({'error': 'Invalid color space'}), 400

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", converted_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_histogram_equalization', methods=['POST'])
def apply_histogram_equalization():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", equalized_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_image_rotation', methods=['POST'])
def apply_image_rotation():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    angle = float(request.form['angle'])
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", rotated_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_morphological_transformation', methods=['POST'])
def apply_morphological_transformation():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    if 'operation' not in request.form or 'kernel_size' not in request.form:
        return jsonify({'error': 'Operation or kernel size not provided'}), 400

    image_file = request.files['image']
    operation = request.form['operation']
    kernel_size = int(request.form['kernel_size'])

    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == 'dilation':
        transformed_image = cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erosion':
        transformed_image = cv2.erode(image, kernel, iterations=1)
    elif operation == 'opening':
        transformed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        transformed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        return jsonify({'error': 'Invalid operation'}), 400

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", transformed_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_inverse_color', methods=['POST'])
def apply_inverse_color():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Inverse color transformation
    inverted_image = cv2.bitwise_not(image)

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", inverted_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_color_enhancement', methods=['POST'])
def apply_color_enhancement():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    hue_scalar = float(request.form.get('hue_scalar', 1.0))
    saturation_scalar = float(request.form.get('saturation_scalar', 1.0))
    value_scalar = float(request.form.get('value_scalar', 1.0))

    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels
    h, s, v = cv2.split(hsv_image)

    # Enhance the hue, saturation, and value channels
    h = np.clip(h * hue_scalar, 0, 255).astype(np.uint8)
    s = np.clip(s * saturation_scalar, 0, 255).astype(np.uint8)
    v = np.clip(v * value_scalar, 0, 255).astype(np.uint8)

    # Merge the channels back
    enhanced_hsv_image = cv2.merge([h, s, v])

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", enhanced_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/apply_sharpening', methods=['POST'])
def apply_sharpening():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']
    kernel_size = int(request.form.get('kernel_size', 3))

    # Make sure kernel_size is odd and greater than 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        kernel_size = 3

    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Create a sharpening kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, kernel_size // 2] = 2.0
    kernel -= np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", sharpened_image)
    img_io.write(buffer)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
