import tensorflow as tf
from keras_preprocessing import image

def predict_score_for_image(model, image_path):
    img = image.load_img(image_path, target_size=(512, 384))

    img_array = image.img_to_array(img)
    img_array /= 255.0

    img_tensor = tf.expand_dims(img_array, axis=0)

    score = model.predict(img_tensor, verbose=0)

    return score[0][0]
