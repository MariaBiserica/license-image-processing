import os
import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# def custom_generator(image_generator, labels):
#     for batch in image_generator:
#         idx = (image_generator.batch_index - 1) * image_generator.batch_size
#         yield batch, labels[idx: idx + image_generator.batch_size]
def custom_generator(image_generator, labels):
    while True:
        batch_imgs, batch_idx = next(image_generator), image_generator.batch_index
        start_idx = (batch_idx - 1) * image_generator.batch_size
        end_idx = start_idx + batch_imgs.shape[0]  # Ajustează pentru ultimul batch
        batch_labels = labels[start_idx:end_idx]
        yield batch_imgs, batch_labels


# K.mean(...) calculează media acestor valori, ceea ce reprezintă acuratețea
def custom_accuracy(y_true, y_pred, threshold=0.1):
    return K.mean(K.cast(K.abs(y_true - y_pred) < threshold, 'float32'))


def build_and_train_model(train_dir, val_dir, train_labels_file, val_labels_file, batch_size, epochs):
    # Definirea modelului de bază (fără straturile superioare) pre-antrenat folosind greutăți de la 'ImageNet'
    print("Construim modelul VGG16...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Adăugarea de noi straturi Dense pentru regresie
    x = Flatten()(base_model.output)  # Strat aplatizare output 3D al rețelei într-un vector 1D
    x = Dense(256, activation='relu')(x)  # 256 neuroni
    output = Dense(1, activation='linear')(x)  # Un singur neuron pentru outputul scalar (scorul de calitate)
    vgg16_model = Model(inputs=base_model.input, outputs=output)

    # Congelarea straturilor din modelul de bază pentru a păstra proprietățile modelului mai complex
    for layer in base_model.layers:
        layer.trainable = False

    print("Compilăm modelul...")
    vgg16_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='mean_squared_error', metrics=[custom_accuracy])

    vgg16_model.summary()

    # Callback pentru salvarea celui mai bun model
    checkpoint = ModelCheckpoint('../../../model_new/best_model.h5',
                                 monitor='val_loss', mode='min', save_best_only=True)

    # Configurarea ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)  # Adăugare normalizare ca preprocesare
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Crearea Generatoarelor de Date folosind 'flow_from_directory'
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=batch_size, class_mode=None)
    validation_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=batch_size, class_mode=None)

    # Citirea etichetelor din fișierele CSV
    train_labels = pd.read_csv(train_labels_file)['MOS'].values
    val_labels = pd.read_csv(val_labels_file)['MOS'].values

    train_custom_generator = custom_generator(train_generator, train_labels)
    validation_custom_generator = custom_generator(validation_generator, val_labels)

    print("Începem antrenamentul modelului...")
    vgg16_model.fit(
        train_custom_generator,
        steps_per_epoch=train_generator.samples // 16,
        epochs=epochs,
        validation_data=validation_custom_generator,
        validation_steps=validation_generator.samples // 16,
        callbacks=[checkpoint, tensorboard_callback])

    print("Evaluăm modelul pe setul de validare...")
    val_loss = vgg16_model.evaluate(
        validation_custom_generator,
        steps=validation_generator.samples // 16)
    print(f'MSE Loss on validation set: {val_loss}')

    return vgg16_model


if __name__ == "__main__":
    # histogram_freq=1 va scrie histograma gradientilor și a ponderilor pentru fiecare epoca
    tensorboard_callback = TensorBoard(log_dir='../logs_new', histogram_freq=1)

    train_directory = '../data/train'
    val_directory = '../data/validation'
    train_lb = '../data/train_labels.csv'
    val_lb = '../data/val_labels.csv'

    print("Antrenăm modelul VGG16...")
    model = build_and_train_model(train_directory, val_directory, train_lb, val_lb, 4, 2)

    # Salvează modelul pentru a fi folosit în predicții
    print("Salvăm modelul antrenat...")
    model.save('../../../model_new/vgg16_model.h5')
    print("Modelul a fost salvat cu succes.")
