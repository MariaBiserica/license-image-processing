from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from repo.assessment_features.noise_assessment.noise_level_assessment import load_elm_model, calculate_noise_score
from repo.assessment_features.contrast_assessment.contrast_level_assessment import calculate_scaled_contrast_score

app = Flask(__name__)

# Directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to the CSV file with contrast scores for scaling
CSV_PATH = '../assessment_features/contrast_assessment/Koniq10k_contrast_scores.csv'

# Load your model (adjust 'elm_model.joblib' as needed)
elm_model = load_elm_model('../assessment_features/noise_assessment/elm_model.joblib')


@app.route('/predict', methods=['POST'])
def predict_quality():
    print("Received request for quality prediction")

    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Use the loaded image for both noise and contrast assessment
    image = cv2.imread(file_path)
    noise_score = calculate_noise_score(elm_model, image)

    # Calculate the contrast score and scaled contrast score
    contrast_score = calculate_scaled_contrast_score(file_path, CSV_PATH)

    # Clean up the uploaded image after processing
    os.remove(file_path)

    # Return the quality score along with contrast scores
    return jsonify({
        'noise_score': noise_score[0],
        'contrast_score': contrast_score
    })


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
