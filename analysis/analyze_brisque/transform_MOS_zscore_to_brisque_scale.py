import pandas as pd

# Define the input and output CSV file paths
input_csv_path = 'output_scores_Koniq10k_original_scale.csv'  # Replace with your actual input file path
output_csv_path = 'output_scores_Koniq10k_brisque_original_scale.csv'  # Replace with your desired output file path

# Read the input CSV file
df = pd.read_csv(input_csv_path)

# Perform the transformation: 100 - MOS_zscore
df['MOS_zscore_transformed_to_brisque_scale'] = 100 - df['MOS_zscore']

# Select the relevant columns
output_df = df[['image_name', 'brisque_score', 'MOS_zscore_transformed_to_brisque_scale']]

# Save the result to a new CSV file
output_df.to_csv(output_csv_path, index=False)

print(f'Transformed scores have been saved to {output_csv_path}')
