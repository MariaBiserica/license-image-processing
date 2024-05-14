import time

import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from repo.alternate_VGG16.utils.metrics import rmse, srocc, plcc, custom_accuracy


# Function to predict the quality of the image using the loaded model
def predict_image_quality(model, image_path):
    new_image = Image.open(image_path)
    new_image = new_image.resize((384, 512))
    new_image = img_to_array(new_image)
    new_image /= 255.0  # normalize - so the pixels are between the values 0-1
    new_image = np.expand_dims(new_image, axis=0)
    predicted_quality = model.predict(new_image)
    predicted_quality = np.clip(predicted_quality, 1, 5)
    return predicted_quality[0][0]


def measure_vgg16(img_path):
    start_time = time.time()  # Start timer

    # Load the trained model with custom objects
    model = load_model('../../alternate_model_Koniq10k_nofreeze_huber_100/model_de_test.h5',
                       custom_objects={'custom_accuracy': custom_accuracy, 'srocc': srocc, 'plcc': plcc, 'rmse': rmse})

    # Predict the quality score
    predicted_quality_score = predict_image_quality(model, img_path)

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute duration

    return predicted_quality_score, f"{elapsed_time:.4f} s"  # Return score and time taken


def main():
    image_path = "data\\512x384\\826373.jpg"
    predicted_quality_score = measure_vgg16(image_path)
    print(f'VGG16 Predicted Quality Score: {predicted_quality_score:.4f}')


if __name__ == "__main__":
    main()
