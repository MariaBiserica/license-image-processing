import os

import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import Huber

from repo.alternate_VGG16.utils.metrics import rmse, srocc, plcc, custom_accuracy

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def build_model_architecture():
    # Definirea modelului de bază (fără straturile superioare) pre-antrenat folosind greutăți de la 'ImageNet'
    print("Construim modelul VGG16...")
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(512, 384, 3))

    # Congelarea straturilor din modelul de bază pentru a păstra proprietățile modelului mai complex
    # for layer in base_model.layers:
    #     layer.trainable = False

    # Adăugarea de noi straturi Dense pentru regresie
    x = Flatten()(base_model.output)  # Strat aplatizare output 3D al rețelei într-un vector 1D
    x = Dense(128, activation='relu')(x)  # 128 neuroni
    x = Dropout(0.2)(x)  # Adăugare strat cu o rată de dropout de 0.2
    output = Dense(1, activation='linear')(x)  # Un singur neuron pentru output-ul scalar (scorul de calitate)
    vgg16_model = Model(inputs=base_model.input, outputs=output)

    return vgg16_model


def train_model(train_dir, val_dir, train_labels_file, val_labels_file, batch_size, epochs, weights):
    vgg16_model = build_model_architecture()

    # Conditional loading of weights if a path is provided and the file exists
    if weights and os.path.exists(weights):
        print(f"Loading weights from {weights}")
        vgg16_model.load_weights(weights)
    else:
        print("Starting training from scratch.")

    print("Compilăm modelul...")
    vgg16_model.compile(optimizer=Adam(learning_rate=1e-4),
                        loss=Huber(),
                        metrics=[custom_accuracy, srocc, plcc, rmse, 'mae'])

    vgg16_model.summary()

    # Save the best model based on validation loss
    checkpoint_best = ModelCheckpoint(
        '../../../alternate_model/best_model.h5',
        monitor='val_loss', save_best_only=True, verbose=1)

    # Save weights every epoch
    checkpoint_epoch = ModelCheckpoint(
        '../../../alternate_model/weights_epoch_{epoch:02d}.h5',
        save_weights_only=True, save_freq='epoch', verbose=1)

    # Include both callbacks in your model's fit method
    callbacks_list = [checkpoint_best, checkpoint_epoch, tensorboard_callback]

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
        callbacks=callbacks_list)

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

    # weights_path = '../../../alternate_model/previous_weights1/weights_epoch_21.h5'
    weights_path = ''
    train_directory = '../Koniq10k/train/all_classes'
    val_directory = '../Koniq10k/validation/all_classes'
    train_lb = '../Koniq10k/train_labels.csv'
    val_lb = '../Koniq10k/val_labels.csv'

    print("Antrenăm modelul...")
    batch: int = 16
    epoch: int = 130  # left epochs
    model = train_model(train_directory, val_directory, train_lb, val_lb, batch, epoch, weights_path)

    test_directory = '../Koniq10k/test/all_classes'
    test_lb = '../Koniq10k/test_labels.csv'

    print("Evaluăm modelul pe setul de testare...")
    evaluate_model(model, test_directory, test_lb, batch)

    # Salvează modelul pentru a fi folosit în predicţii
    print("Salvăm modelul antrenat...")
    model.save('../../../alternate_model/model_de_test.h5')
    print("Modelul a fost salvat cu succes.")
