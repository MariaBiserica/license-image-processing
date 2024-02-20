import os

import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from repo.alternate_VGG16.utils.metrics import srocc, plcc, custom_accuracy

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def build_and_train_model(train_dir, val_dir, train_labels_file, val_labels_file, batch_size, epochs):
    # Definirea modelului de bază (fără straturile superioare) pre-antrenat folosind greutăți de la 'ImageNet'
    print("Construim modelul VGG16...")
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(512, 384, 3))

    # Congelarea straturilor din modelul de bază pentru a păstra proprietățile modelului mai complex
    for layer in base_model.layers:
        layer.trainable = False

    # Adăugarea de noi straturi Dense pentru regresie
    x = Flatten()(base_model.output)  # Strat aplatizare output 3D al rețelei într-un vector 1D
    x = Dense(128, activation='relu')(x)  # 128 neuroni
    x = Dropout(0.2)(x)  # Adăugare strat cu o rată de dropout de 0.2
    output = Dense(1, activation='linear')(x)  # Un singur neuron pentru output-ul scalar (scorul de calitate)
    vgg16_model = Model(inputs=base_model.input, outputs=output)

    print("Compilăm modelul...")
    vgg16_model.compile(optimizer=Adam(learning_rate=1e-4),
                        loss='mse',
                        metrics=['mae', srocc, plcc, custom_accuracy])

    vgg16_model.summary()

    # Callback pentru salvarea celui mai bun model
    checkpoint = ModelCheckpoint('../../../alternate_model_without_resized_data/best_model.h5',
                                 monitor='val_loss', save_best_only=True)

    # Configurarea ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255)  # Adăugare normalizare ca preprocesare
    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                     horizontal_flip=True)

    # Citirea etichetelor din fișierele CSV
    train_df = pd.read_csv(train_labels_file)
    val_df = pd.read_csv(val_labels_file)

    # Crearea Generatoarelor de Date folosind 'flow_from_dataframe'
    train_generator = train_datagen.flow_from_dataframe(
        train_df, directory=train_dir, x_col='image_name', y_col='MOS',
        target_size=(512, 384), batch_size=batch_size, class_mode='raw')

    validation_generator = val_datagen.flow_from_dataframe(
        val_df, directory=val_dir, x_col='image_name', y_col='MOS',
        target_size=(512, 384), batch_size=batch_size, class_mode='raw')

    print("Începem antrenamentul modelului...")
    vgg16_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, tensorboard_callback])

    return vgg16_model


def evaluate_model(vgg16_model, test_dir, test_labels_file, batch_size):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Citirea etichetelor din fișierele CSV
    test_df = pd.read_csv(test_labels_file)

    # Crearea unui generator de date folosind 'flow_from_dataframe'
    test_generator = test_datagen.flow_from_dataframe(
        test_df, directory=test_dir, x_col='image_name', y_col='MOS',
        target_size=(512, 384), batch_size=batch_size, class_mode='raw')

    values = vgg16_model.evaluate(
        test_generator,
        steps=test_generator.samples // batch_size)
    print(f'Values (mse, mae, srocc, plcc, custom_accuracy): {values}')


if __name__ == "__main__":
    # histogram_freq=1 va scrie histograma gradientilor și a ponderilor pentru fiecare epoca
    tensorboard_callback = TensorBoard(log_dir='../logs_new', histogram_freq=1)

    train_directory = '../data/train/all_classes'
    val_directory = '../data/validation/all_classes'
    train_lb = '../data/train_labels.csv'
    val_lb = '../data/val_labels.csv'

    print("Antrenăm modelul...")
    batch: int = 16
    epoch: int = 40
    model = build_and_train_model(train_directory, val_directory, train_lb, val_lb, batch, epoch)

    test_directory = '../data/test/all_classes'
    test_lb = '../data/test_labels.csv'

    print("Evaluăm modelul pe setul de testare...")
    evaluate_model(model, test_directory, test_lb, batch)

    # Salvează modelul pentru a fi folosit în predicţii
    print("Salvăm modelul antrenat...")
    model.save('../../../alternate_model_without_resized_data/model_de_test.h5')
    print("Modelul a fost salvat cu succes.")
