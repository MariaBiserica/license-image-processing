import time
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
from scipy.stats import spearmanr


def custom_accuracy(y_true, y_pred, threshold=0.1):
    return K.mean(K.cast(K.abs(y_true - y_pred) < threshold, 'float32'))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def srocc(y_true, y_pred):
    def compute_srocc(y_true, y_pred):
        srocc_value, _ = spearmanr(y_true, y_pred)
        return srocc_value

    srocc_value = tf.py_function(compute_srocc, [y_true, y_pred], tf.float32)
    return srocc_value


def plcc(y_true, y_pred):
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred
    numerator = K.sum(centered_true * centered_pred)
    denominator = K.sqrt(K.sum(K.square(centered_true)) * K.sum(K.square(centered_pred)))
    plcc_value = numerator / denominator
    return plcc_value


# Function to predict the quality of the image using the loaded model
def predict_image_quality(model, image_path):
    new_image = Image.open(image_path)
    # de vazut daca trebuie resize sau daca pot sa iau pe bucati daca e imaginea mai mare sau sa fac crop
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
    model = load_model('../alternate_model_Koniq10k_nofreeze_huber_100/model_de_test.h5',
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
