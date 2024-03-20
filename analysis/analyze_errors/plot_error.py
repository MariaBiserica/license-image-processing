import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from repo.alternate_VGG16.utils.metrics import rmse, srocc, plcc, custom_accuracy
from repo.analysis.analyze_errors.predict_score_for_image import predict_score_for_image


def predict_scores_for_images(file_path, image_dir, image_names):
    model = load_model(file_path,
                       custom_objects={
                           'custom_accuracy': custom_accuracy,
                           'rmse': rmse,
                           'srocc': srocc,
                           'plcc': plcc
                       })

    predicted_scores = []
    for j, image_name in enumerate(image_names):
        image_path = image_dir + image_name

        # Print current image processing status
        print(f"Processing {j + 1}/{len(image_names)}: {image_name}")

        predicted_score = predict_score_for_image(model, image_path)  # Predict score for the image
        predicted_scores.append(predicted_score)
    return predicted_scores


if __name__ == "__main__":
    data_directory = '../../alternate_VGG16/LIVE2/'
    test_images = data_directory + 'test/all_classes/'
    test_scores = data_directory + 'test_labels.csv'

    df = pd.read_csv(test_scores)

    image_names = df['image_name'].values
    true_scores = df['MOS'].values

    file_path = '../../../alternate_model_LIVE2_nofreeze_huber_100/best_model.h5'
    predicted_scores = predict_scores_for_images(file_path, test_images, image_names)
    predicted_scores = np.array(predicted_scores)

    # Calculate error for each image
    errors = predicted_scores - true_scores

    # Plot the results
    plt.scatter(true_scores, errors)
    plt.xlabel('True Scores')
    plt.ylabel('Error')
    plt.title('Difference (pred - true) vs. True Scores')
    plt.show()

    # Print some example predicted and true scores along with errors
    for i in range(5):
        print(
            f"Image Name: {image_names[i]}, "
            f"Predicted Score: {predicted_scores[i]}, "
            f"True Score: {true_scores[i]}, "
            f"Error: {errors[i]}")
