import os
import pandas as pd
from repo.brisque_release_online.brisque_master.brisque.brisque_quality import measure_brisque as brisque_score


# Function to prepare and save the valid images and their scores
def save_valid_images_with_scores(image_folder_path, csv_file_path, output_csv_path):
    # Load the ground truth CSV
    gt_df = pd.read_csv(csv_file_path)

    # Add a column to the dataframe for BRISQUE scores
    gt_df['brisque_score'] = gt_df['image_name'].apply(lambda x: brisque_score(os.path.join(image_folder_path, x)))

    # Filter out images with valid BRISQUE scores and corresponding ground truth scores
    valid_scores_df = gt_df[(gt_df['brisque_score'] >= 0) & (gt_df['brisque_score'] <= 100)]

    # Select the relevant columns to save to the CSV
    valid_scores_df = valid_scores_df[['image_name', 'brisque_score', 'MOS_zscore']]

    # Save to the output CSV file
    valid_scores_df.to_csv(output_csv_path, index=False)
    print(f'Valid images with scores have been saved to {output_csv_path}')


# Define the paths
image_folder_path = '../../VGG16/data/512x384'
csv_file_path = '../../VGG16/data/koniq10k_scores_and_distributions.csv'
output_csv_path = 'output_scores_Koniq10k_original_scale.csv'

# Call the function to save the data
save_valid_images_with_scores(image_folder_path, csv_file_path, output_csv_path)
