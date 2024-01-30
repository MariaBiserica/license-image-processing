import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split


def create_train_val_directories_and_labels(base_image_dir, csv_file, train_dir, val_dir,
                                            train_labels_file, val_labels_file, test_size=0.1):
    # Citirea etichetelor din fișierul CSV
    df = pd.read_csv(csv_file)

    # Împărțirea datelor în seturi de antrenament și validare
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    # Copierea imaginilor și salvarea etichetelor
    train_df.to_csv(train_labels_file, index=False)
    val_df.to_csv(val_labels_file, index=False)

    for _, row in train_df.iterrows():
        shutil.copy(os.path.join(base_image_dir, row['image_name']), train_dir)

    for _, row in val_df.iterrows():
        shutil.copy(os.path.join(base_image_dir, row['image_name']), val_dir)

    print("Procesul de copiere a fost finalizat.")


# Setarea căilor și a parametrilor
base_image_directory = '../data/512x384'  # Directorul unde sunt stocate toate imaginile
csv = '../data/koniq10k_scores_and_distributions.csv'  # Calea fișierului CSV
train_directory = '../data/train'  # Directorul unde vor fi stocate imaginile de antrenament
val_directory = '../data/validation'  # Directorul unde vor fi stocate imaginile de validare
train_labels = '../data/train_labels.csv'
val_labels = '../data/val_labels.csv'

create_train_val_directories_and_labels(base_image_directory, csv, train_directory, val_directory,
                                        train_labels, val_labels)
