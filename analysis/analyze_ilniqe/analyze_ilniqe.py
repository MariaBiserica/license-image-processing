import os
import csv
from repo.ilniqe_release_online.ilniqe_master.ilniqe \
    import measure_ilniqe as ilniqe_score

# Path to the image folder - we will use only the images with a valid brisque score
image_folder_path = '../analyze_brisque/valid_images_Koniq10k'

# Path and name for the output CSV file
output_csv_path = 'ilniqe_scores_Koniq10k.csv'

# Get all image filenames in the folder
image_filenames = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_filenames)

# Open the CSV file for writing
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'ilniqe_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over the images in the folder
    for index, filename in enumerate(image_filenames, start=1):
        full_path = os.path.join(image_folder_path, filename)
        score = ilniqe_score(full_path)
        writer.writerow({'image_name': filename, 'ilniqe_score': score})

        # Print the progress
        print(f"Processing {index}/{total_images}: {filename}")

print("IL-NIQE scores calculation completed and results are saved to", output_csv_path)
