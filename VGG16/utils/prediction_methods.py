import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array


# Prediction by classic resize
def predict_image_quality(model, image_path):
    new_image = Image.open(image_path)
    new_image = new_image.resize((384, 512))
    new_image = img_to_array(new_image)
    new_image /= 255.0  # normalize - so the pixels are between the values 0-1
    new_image = np.expand_dims(new_image, axis=0)
    predicted_quality = model.predict(new_image)
    predicted_quality = np.clip(predicted_quality, 1, 5)
    return predicted_quality[0][0]


# Prediction by cropping central fragment
def predict_image_quality_central_crop(model, image_path):
    new_image = Image.open(image_path)
    width, height = new_image.size
    target1, target2 = 384, 512

    # Calculate start points to crop the central fragment
    start_x = width // 2 - target1 // 2
    start_y = height // 2 - target2 // 2

    if width > target2 or height > target1:
        # Crop the central fragment
        print("Cropping...")
        fragment = new_image.crop((start_x, start_y, start_x + target1, start_y + target2))
    else:
        # Resize the image if it is smaller than the target dimensions
        fragment = new_image.resize((target1, target2))

    # Process the fragment
    fragment = img_to_array(fragment)
    fragment /= 255.0  # normalize
    fragment = np.expand_dims(fragment, axis=0)

    # Predict the quality
    predicted_quality = model.predict(fragment)
    final_score = predicted_quality[0][0]

    # Clip the score to be within the expected range
    final_score = np.clip(final_score, 1, 5)
    return final_score


# Predict by cropping multiple overlapping fragments calculate mean score
def predict_image_quality_fragments(model, image_path):
    new_image = Image.open(image_path)
    width, height = new_image.size
    target1, target2 = 384, 512
    if width > target2 or height > target1:
        # Assuming square fragments for simplicity
        scores = []
        for i in range(0, height, target2 // 2):
            for j in range(0, width, target1 // 2):
                fragment = new_image.crop((j, i, j + target1, i + target2))
                fragment = fragment.resize((target1, target2))
                fragment = img_to_array(fragment)
                fragment /= 255.0
                fragment = np.expand_dims(fragment, axis=0)
                predicted_quality = model.predict(fragment)
                scores.append(predicted_quality[0][0])
        final_score = np.mean(scores)  # Average of all fragment scores
    else:
        new_image = new_image.resize((target1, target2))
        new_image = img_to_array(new_image)
        new_image /= 255.0
        new_image = np.expand_dims(new_image, axis=0)
        predicted_quality = model.predict(new_image)
        final_score = predicted_quality[0][0]

    final_score = np.clip(final_score, 1, 5)
    return final_score


# Prediction by resizing and adaptive cropping
def predict_image_quality_resizing_and_adaptive_cropping(model, image_path):
    def resize_and_crop(image, target_size):
        width, height = image.size
        target_width, target_height = target_size

        if width <= target_width and height <= target_height:
            # If image is smaller or equal to the target size, just resize
            return image.resize(target_size, Image.ANTIALIAS)

        # Resize the image maintaining the aspect ratio
        aspect_ratio = min(target_width / width, target_height / height)
        new_size = (int(width * aspect_ratio), int(height * aspect_ratio))
        resized_image = image.resize(new_size, Image.ANTIALIAS)

        # Calculate the coordinates for cropping the central region
        left = (resized_image.width - target_width) // 2
        top = (resized_image.height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        # Crop the central region
        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image

    new_image = Image.open(image_path)
    target_size = (384, 512)

    # Check the size of the image
    if new_image.size[0] > target_size[0] or new_image.size[1] > target_size[1]:
        # Resize and crop the image if it is larger than the target size
        processed_image = resize_and_crop(new_image, target_size)
    else:
        # Directly resize the image if it is smaller than or equal to the target size
        processed_image = new_image.resize(target_size, Image.ANTIALIAS)

    # Convert to array and normalize
    processed_image = img_to_array(processed_image)
    processed_image /= 255.0  # normalize
    processed_image = np.expand_dims(processed_image, axis=0)

    # Predict the quality
    predicted_quality = model.predict(processed_image)
    final_score = predicted_quality[0][0]

    # Clip the score to be within the expected range
    final_score = np.clip(final_score, 1, 5)
    return final_score
