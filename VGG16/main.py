from keras.models import load_model
from utils.predictions import predict_image_quality

if __name__ == "__main__":
    model = load_model('../../models_vgg16/vgg16_model.h5')
    image_path = "..\\ilniqe_release_online\\IL-NIQE-master\\pepper_exa\\pepper_4.png"
    predicted_score = predict_image_quality(model, image_path)
    print(f'Predicted Quality Score: {predicted_score:.4f}')

