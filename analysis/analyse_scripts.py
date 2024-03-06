import os
import pandas as pd
import numpy as np

# from repo.brisque_release_learnopencv.Python.libsvm.python.brisquequality \
#     import measure_brisque as brisque_learnopencv_score
from repo.brisque_release_online.brisque_master.brisque.brisque_quality \
    import measure_brisque as brisque_score
# from repo.ilniqe_release_online.ilniqe_master.ilniqe \
#     import measure_ilniqe as ilniqe_score
from repo.niqe_release_online.niqe import measure_niqe as niqe_score
# from repo.VGG16.vgg16_quality_score import measure_vgg16 as vgg16_score
from repo.analysis.performance_metrics import srcc, plcc, rmse, mae


def calculate_metrics(predicted_scores, ground_truth_scores):
    """Calculate SRCC, PLCC, RMSE, and MAE for a batch."""
    srcc_value = srcc(np.array(predicted_scores), np.array(ground_truth_scores))
    plcc_value = plcc(np.array(predicted_scores), np.array(ground_truth_scores))
    rmse_value = rmse(np.array(predicted_scores), np.array(ground_truth_scores))
    mae_value = mae(np.array(predicted_scores), np.array(ground_truth_scores))
    return srcc_value, plcc_value, rmse_value, mae_value


def main():
    csv_file_path = '..\\VGG16\\data\\koniq10k_scores_and_distributions.csv'
    df = pd.read_csv(csv_file_path)
    # mos_zscore_mapping = dict(zip(df['image_name'], df['MOS_zscore']))
    mos_mapping = dict(zip(df['image_name'], df['MOS']))

    # Define the folder containing the images
    image_folder_path = "..\\VGG16\\data\\512x384"
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # File to store the results
    output_file_path = 'complete_image_quality_analysis.txt'
    # header = "Image Name, BRISQUE Score, IL-NIQE Score, NIQE Score, VGG16 Score, GROUND TRUTH (MOS Z-Score) \n"
    header = "Image Name, BRISQUE Score, NIQE Score, GROUND TRUTH (MOS) \n"

    # Iterate over each image in the folder
    batch_size = 16
    total_batches = len(image_files) // batch_size + (1 if len(image_files) % batch_size > 0 else 0)
    with open(output_file_path, 'w') as output_file:
        output_file.write(header)
        batch_number = 0
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_number += 1
            predicted_brisque_scores = []
            predicted_niqe_scores = []
            ground_truth_mos_scores = []

            for index, image_name in enumerate(batch_files):
                image_path = os.path.join(image_folder_path, image_name)
                if image_name in mos_mapping:  # Check if the image has a corresponding MOS value
                    brisque_quality = brisque_score(image_path)
                    # il_niqe_quality = ilniqe_score(image_path)
                    niqe_quality = niqe_score(image_path)
                    # vgg16_quality = vgg16_score(image_path)

                    # mos_zscore = mos_zscore_mapping.get(image_name, 'N/A')
                    mos_score = mos_mapping.get(image_name, np.nan)

                    # Print progress
                    print(f"Processing image {i + index + 1}/{len(image_files)}, Batch {batch_number}/{total_batches}: {image_name}")

                    # Write the results for each image to th output file
                    output_file.write(f"{image_name}, {brisque_quality:.4f}, {niqe_quality:.4f}, {mos_score:.4f}\n")

                    # Collect predicted and GT scores
                    predicted_brisque_scores.append(brisque_quality)
                    predicted_niqe_scores.append(niqe_quality)
                    ground_truth_mos_scores.append(mos_score)

            # Calculate metrics for BRISQUE and NIQE
            srcc_brisque, plcc_brisque, rmse_brisque, mae_brisque = calculate_metrics(predicted_brisque_scores, ground_truth_mos_scores)
            srcc_niqe, plcc_niqe, rmse_niqe, mae_niqe = calculate_metrics(predicted_niqe_scores, ground_truth_mos_scores)

            # Write metrics
            metrics_header = "\nBatch Number: {}, Metrics\n".format(batch_number)
            metrics_header += "Metric, BRISQUE, NIQE\n"
            metrics_table = f"SRCC, {srcc_brisque:.4f}, {srcc_niqe:.4f}\n"
            metrics_table += f"PLCC, {plcc_brisque:.4f}, {plcc_niqe:.4f}\n"
            metrics_table += f"RMSE, {rmse_brisque:.4f}, {rmse_niqe:.4f}\n"
            metrics_table += f"MAE, {mae_brisque:.4f}, {mae_niqe:.4f}\n"

            output_file.write(metrics_header + metrics_table + "\n")


if __name__ == "__main__":
    main()
