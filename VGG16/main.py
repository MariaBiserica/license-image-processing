from keras.models import load_model
from utils.predictions import predict_image_quality
from keras import backend as K


def custom_accuracy(y_true, y_pred, threshold=0.1):
    return K.mean(K.cast(K.abs(y_true - y_pred) < threshold, 'float32'))


if __name__ == "__main__":
    model = load_model('../../models_vgg16/vgg16_model.h5', custom_objects={'custom_accuracy': custom_accuracy})
    image_path = "..\\ilniqe_release_online\\IL-NIQE-master\\pepper_exa\\bikes.bmp"
    predicted_score = predict_image_quality(model, image_path)
    print(f'Predicted Quality Score: {predicted_score:.4f}')

