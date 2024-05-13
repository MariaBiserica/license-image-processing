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
        results['noise_score'] = calculate_scaled_noise_score(elm_model, image, NOISE_CSV_PATH)
    if 'Contrast' in selected_metrics:
        results['contrast_score'] = calculate_scaled_contrast_score(file_path, CONTRAST_CSV_PATH)
    if 'Brightness' in selected_metrics:
        results['brightness_score'] = calculate_scaled_brightness_score(file_path, BRIGHTNESS_CSV_PATH)
    if 'Sharpness' in selected_metrics:
        results['sharpness_score'] = calculate_scaled_sharpness_score(file_path, SHARPNESS_CSV_PATH)
    if 'Chromatic Quality' in selected_metrics:
        results['chromatic_score'] = calculate_scaled_chromatic_score(file_path, CHROMATIC_CSV_PATH, SVR_MODEL_PATH)

    # Clean up the uploaded image after processing
    os.remove(file_path)

    # Return the calculated scores
    return jsonify(results)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
