import os
import pandas as pd

# from repo.brisque_release_learnopencv.Python.libsvm.python.brisquequality \
#     import measure_brisque as brisque_learnopencv_score
from repo.brisque_release_online.brisque_master.brisque.brisque_quality \
    import measure_brisque as brisque_score
from repo.ilniqe_release_online.ilniqe_master.ilniqe \
    import measure_ilniqe as ilniqe_score
from repo.niqe_release_online.niqe import measure_niqe as niqe_score
from repo.VGG16.vgg16_quality_score import measure_vgg16 as vgg16_score


def main():
    # Load the CSV file to create a mapping of image names to 'MOS_zscore'
    csv_file_path = '..\\VGG16\\data\\koniq10k_scores_and_distributions.csv'
    df = pd.read_csv(csv_file_path)
    mos_zscore_mapping = dict(zip(df['image_name'], df['MOS_zscore']))

    # Define the folder containing the images
    image_folder_path = "..\\VGG16\\data\\512x384"
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # File to store the results
    output_file_path = 'image_quality_analysis.txt'

    # Write header to the output file
    header = "Image Name, BRISQUE Score, IL-NIQE Score, NIQE Score, VGG16 Score, GROUND TRUTH (MOS Z-Score) \n"
    # header = "Image Name, BRISQUE Score, NIQE Score, VGG16 Score, GROUND TRUTH (MOS Z-Score) \n"

    # Iterate over each image in the folder
    with open(output_file_path, 'w') as output_file:
        output_file.write(header)
        for index, image_name in enumerate(image_files, 1):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
                image_path = os.path.join(image_folder_path, image_name)

                # Compute quality scores using the provided scripts
                # brisque_quality_learnopencv = brisque_learnopencv_score(image_path)
                brisque_quality = brisque_score(image_path)
                il_niqe = ilniqe_score(image_path)
                niqe = niqe_score(image_path)
                vgg16_quality = vgg16_score(image_path)

                # Fetch the 'MOS_zscore' for the image from the CSV file
                mos_zscore = mos_zscore_mapping.get(image_name, 'N/A')

                # Write the results to the output file
                output_file.write(f"{image_name}, {brisque_quality}, {il_niqe}, {niqe}, {vgg16_quality}, {mos_zscore}\n")
                # output_file.write(f"{image_name}, {brisque_quality}, {niqe}, {vgg16_quality}, {mos_zscore}\n")

                # Print the progress
                print(f"Processing {index}/{len(image_files)}: {image_name}")


if __name__ == "__main__":
    main()
