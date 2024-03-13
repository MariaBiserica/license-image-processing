import os
import csv
from repo.niqe_release_online.niqe import measure_niqe as niqe_score

# Path to the image folder - we will use only the images with a valid brisque score
image_folder_path = '../analyze_brisque/valid_images_LIVE2'

# Path and name for the output CSV file
output_csv_path = 'niqe_scores_LIVE2.csv'

# Get all image filenames in the folder
image_filenames = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_filenames)

# Open the CSV file for writing
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'niqe_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over the images in the folder
    for index, filename in enumerate(image_filenames, start=1):
        full_path = os.path.join(image_folder_path, filename)
        score = niqe_score(full_path)
        writer.writerow({'image_name': filename, 'niqe_score': score})

        # Print the progress
        print(f"Processing {index}/{total_images}: {filename}")

print("NIQE scores calculation completed and results are saved to", output_csv_path)
