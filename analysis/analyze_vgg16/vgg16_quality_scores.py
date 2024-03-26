import csv
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from repo.alternate_VGG16.utils.metrics import rmse, srocc, plcc, custom_accuracy


def predict_image_quality(model, image_path):
    new_image = Image.open(image_path)
    new_image = new_image.resize((384, 512))
    new_image = img_to_array(new_image)
    new_image /= 255.0  # normalize - so the pixels are between the values 0-1
    new_image = np.expand_dims(new_image, axis=0)
    predicted_quality = model.predict(new_image)
    predicted_quality = np.clip(predicted_quality, 1, 5)
    return predicted_quality[0][0]


def measure_vgg16(model, img_path):
    predicted_quality_score = predict_image_quality(model, img_path)
    return predicted_quality_score


def evaluate_dataset(dataset_path):
    model = load_model('../../../alternate_model_LIVE2_nofreeze_huber_100_valid/best_model.h5',
                       custom_objects={
                           'custom_accuracy': custom_accuracy,
                           'rmse': rmse,
                           'srocc': srocc,
                           'plcc': plcc
                       })
    results = []
    for i, img_name in enumerate(os.listdir(dataset_path), start=1):
        img_path = os.path.join(dataset_path, img_name)
        # Open the image to check its mode
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    print(f"Skipping non-RGB image {img_name}")
                    continue  # Skip this image and move to the next
        except Exception as e:
            print(f"Error opening image {img_name}: {e}")
            continue

        print(f"Processing image {i}/{len(os.listdir(dataset_path))}: {img_name}")
        predicted_score = measure_vgg16(model, img_path)
        results.append((img_name, predicted_score))

    with open('predicted_scores_on_LIVE2_model_trained_on_LIVE2_valid_new.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'MOS_predicted_score'])
        writer.writerows(results)


def main():
    # dataset_path = "../../VGG16/data/512x384"
    dataset_path = "../../alternate_VGG16/data/LIVE2/databaserelease2/LIVE_all"
    evaluate_dataset(dataset_path)
    print("Evaluation completed and saved to predicted_scores.csv")


if __name__ == "__main__":
    main()
