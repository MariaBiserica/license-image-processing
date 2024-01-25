import os
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from repo.VGG16.utils.data_processing import load_and_process_images, split_data
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# K.mean(...) calculează media acestor valori, ceea ce reprezintă acuratețea
def custom_accuracy(y_true, y_pred, threshold=0.1):
    return K.mean(K.cast(K.abs(y_true - y_pred) < threshold, 'float32'))


def build_and_train_model(train_images, train_labels, val_images, val_labels):
    # Definirea modelului VGG16 fără straturile superioare
    # Am început cu modelul VGG16 pre-antrenat (folosind greutăți de la 'ImageNet') ca bază.
    # Acesta este un model de rețea neurală convoluțională popular pentru procesarea imaginilor.
    print("Construim modelul VGG16...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Adăugarea de noi straturi pentru regresie în loc de clasificare
    # Am eliminat straturile superioare ale VGG16 și le-am înlocuit cu propriile straturi.
    # Acest lucru include un strat 'Flatten' pentru aplatizarea output-ului 3D al rețelei VGG16 într-un vector 1D,
    # urmat de două straturi 'Dense', unul cu 256 de neuroni și activare 'relu',
    # și un altul cu un singur neuron și activare 'linear'.
    # Straturile Dense suplimentare sunt folosite pentru regresie în loc de clasificare.
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='linear')(x)  # Un singur neuron pentru outputul scalar (scorul de calitate)

    vgg16_model = Model(inputs=base_model.input, outputs=output)

    # Congelarea straturilor din modelul VGG16 de bază pentru a nu le antrena
    for layer in base_model.layers:
        layer.trainable = False

    # Compilarea modelului cu MSE
    print("Compilăm modelul...")
    # vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    vgg16_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='mean_squared_error',
                        metrics=[custom_accuracy])

    # Callback pentru salvarea celui mai bun model
    checkpoint = ModelCheckpoint('../../../models_vgg16/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    # Configurarea ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)  # Adăugare normalizare ca preprocesare
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Crearea Generatoarelor de Date folosind 'flow'
    train_generator = train_datagen.flow(
        train_images, train_labels,
        batch_size=2)

    validation_generator = val_datagen.flow(
        val_images, val_labels,
        batch_size=2)

    # Antrenarea modelului folosind fit_generator
    print("Începem antrenamentul modelului...")
    vgg16_model.fit(
        train_generator,
        steps_per_epoch=len(train_images) // 4,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(val_images) // 4,
        callbacks=[checkpoint, tensorboard_callback])

    # Evaluarea modelului pe setul de validare
    print("Evaluăm modelul pe setul de validare...")
    val_loss = vgg16_model.evaluate(validation_generator, steps=len(val_images) // 4)
    print(f'MSE Loss on validation set: {val_loss}')

    return vgg16_model


if __name__ == "__main__":
    # histogram_freq=1 va scrie histograma gradientilor și a ponderilor pentru fiecare epoca
    tensorboard_callback = TensorBoard(log_dir='../logs', histogram_freq=1)

    print("Încărcăm și prelucrăm imaginile...")
    path_to_images = '../data/512x384'
    path_to_csv = '../data/koniq10k_scores_and_distributions.csv'

    images, labels = load_and_process_images(path_to_images, path_to_csv)
    train_images, val_images, train_labels, val_labels = split_data(images, labels)

    print("Antrenăm modelul VGG16...")
    # custom_callback = CustomCallback()
    model = build_and_train_model(train_images, train_labels, val_images, val_labels)

    # Salvează modelul pentru a fi folosit în predicții
    print("Salvăm modelul antrenat...")
    model.save('vgg16_model.h5')
    print("Modelul a fost salvat cu succes.")
