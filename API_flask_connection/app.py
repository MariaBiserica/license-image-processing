from flask import Flask, request, jsonify, json
from werkzeug.utils import secure_filename
import os
import cv2
from repo.assessment_features.noise_assessment.noise_level_assessment import (load_elm_model,
                                                                              calculate_scaled_noise_score)
from repo.assessment_features.contrast_assessment.contrast_level_assessment import calculate_scaled_contrast_score
from repo.assessment_features.brightness_assessment.brightness_level_assessment import calculate_scaled_brightness_score
from repo.assessment_features.sharpness_assessment.sharpness_level_assessment import calculate_scaled_sharpness_score
from repo.assessment_features.chromatic_assessment.chromatic_level_assessment import calculate_scaled_chromatic_score
from repo.brisque_release_online.brisque_master.brisque.brisque_quality import calculate_scaled_brisque_score
from repo.ilniqe_release_online.ilniqe_master.ilniqe import calculate_scaled_ilniqe_score

app = Flask(__name__)

# Directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to the CSV file with contrast scores for scaling
NOISE_CSV_PATH = '../assessment_features/noise_assessment/Koniq10k_noise_scores.csv'
CONTRAST_CSV_PATH = '../assessment_features/contrast_assessment/Koniq10k_contrast_scores.csv'
BRIGHTNESS_CSV_PATH = '../assessment_features/brightness_assessment/Koniq10k_brightness_scores.csv'
SHARPNESS_CSV_PATH = '../assessment_features/sharpness_assessment/Koniq10k_sharpness_scores.csv'
CHROMATIC_CSV_PATH = '../assessment_features/chromatic_assessment/Koniq10k_chromatic_scores.csv'
SVR_MODEL_PATH = '../assessment_features/chromatic_assessment/svr_model.joblib'
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
    if 'ILNIQE' in selected_metrics:
        ilniqe_score, ilniqe_time = calculate_scaled_ilniqe_score(file_path, ILNIQE_SCORES_CSV_PATH)
        results['ilniqe_score'] = f"{ilniqe_score:.4f}"
        results['ilniqe_time'] = ilniqe_time

    # Clean up the uploaded image after processing
    os.remove(file_path)

    # Return the calculated scores
    return jsonify(results)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
