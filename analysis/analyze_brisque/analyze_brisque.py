import os
import pandas as pd
import matplotlib.pyplot as plt
from repo.brisque_release_online.brisque_master.brisque.brisque_quality \
    import measure_brisque as brisque_score
import shutil
import csv
from repo.analysis.scale_brisque_to_mos import transform_score

# Path to the image folder
image_folder_path = '../../alternate_VGG16/data/LIVE2/databaserelease2/LIVE_all'

# Paths and names for the output CSV files
output_csv_path = 'output_scores_LIVE2.csv'
negative_scores_csv_path = 'negative_scores_LIVE2.csv'
big_scores_csv_path = 'big_scores_LIVE2.csv'
na_scores_csv_path = 'na_scores_LIVE2.csv'

# Path to the CSV file with ground truth scores
csv_file_path = '../../alternate_VGG16/data/LIVE2/LIVE2_MOS_scores.csv'
df = pd.read_csv(csv_file_path)
mos_mapping = dict(zip(df['image_name'], df['MOS']))

# Directories for image categories
valid_images_dir = 'valid_images_LIVE2'
negative_score_images_dir = 'negative_score_images_LIVE2'
big_score_images_dir = 'big_score_images_LIVE2'
na_score_images_dir = 'na_score_images_LIVE2'
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(negative_score_images_dir, exist_ok=True)
os.makedirs(big_score_images_dir, exist_ok=True)
os.makedirs(na_score_images_dir, exist_ok=True)

# Score categories count
scores_count = {'<0': 0, '[0-100]': 0, '>100': 0, 'N/A': 0}

# Open the CSV files for writing
with open(output_csv_path, 'w', newline='') as output_csvfile, \
     open(negative_scores_csv_path, 'w', newline='') as negative_csvfile, \
     open(big_scores_csv_path, 'w', newline='') as big_csvfile, \
     open(na_scores_csv_path, 'w', newline='') as na_csvfile:

    output_fieldnames = ['image_name', 'brisque_score', 'ground_truth', 'transformed_mos']
    simple_fieldnames = ['image_name', 'brisque_score']
    output_writer = csv.DictWriter(output_csvfile, fieldnames=output_fieldnames)
    negative_writer = csv.DictWriter(negative_csvfile, fieldnames=simple_fieldnames)
    big_writer = csv.DictWriter(big_csvfile, fieldnames=simple_fieldnames)
    na_writer = csv.DictWriter(na_csvfile, fieldnames=simple_fieldnames)
    output_writer.writeheader()
    negative_writer.writeheader()
    big_writer.writeheader()
    na_writer.writeheader()

    brisque_scores = []

    # Iterate over the images in the folder
    for filename in os.listdir(image_folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(image_folder_path, filename)
            score = brisque_score(full_path)
            brisque_scores.append(score)
            ground_truth = mos_mapping.get(filename, 'N/A')

            transformed_mos = transform_score(score) if ground_truth != 'N/A' else 'N/A'
            output_writer.writerow({'image_name': filename, 'brisque_score': score, 'ground_truth': ground_truth,
                                    'transformed_mos': transformed_mos})

            if score < 0:
                negative_writer.writerow({'image_name': filename, 'brisque_score': score})
                shutil.copy(full_path, os.path.join(negative_score_images_dir, filename))
                scores_count['<0'] += 1
            elif score > 100:
                big_writer.writerow({'image_name': filename, 'brisque_score': score})
                shutil.copy(full_path, os.path.join(big_score_images_dir, filename))
                scores_count['>100'] += 1
            elif ground_truth == 'N/A':
                na_writer.writerow({'image_name': filename, 'brisque_score': score})
                shutil.copy(full_path, os.path.join(na_score_images_dir, filename))
                scores_count['N/A'] += 1
            else:
                shutil.copy(full_path, os.path.join(valid_images_dir, filename))
                scores_count['[0-100]'] += 1

# Create and display the distribution graph of BRISQUE scores
plt.hist(brisque_scores, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of BRISQUE Scores')
plt.xlabel('BRISQUE Score')
plt.ylabel('Number of Images')
plt.show()

# Display a pie chart for score categories
labels = scores_count.keys()
sizes = scores_count.values()
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of Images by Score Category')
plt.show()
