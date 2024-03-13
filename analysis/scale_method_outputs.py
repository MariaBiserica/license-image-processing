import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from repo.analysis.scale_brisque_to_mos import transform_score


def calculate_translation_coefficients(scores, target_min=0, target_max=100):
    original_min = np.min(scores)
    original_max = np.max(scores)
    scale = (target_max - target_min) / (original_max - original_min)
    shift = target_min - original_min * scale
    return scale, shift


def translate_scores(scores, scale, shift):
    return scores * scale + shift


def process_scores(csv_path, output_csv_path):
    # Read the CSV file containing image names and quality scores
    df = pd.read_csv(csv_path)

    # Extract quality scores
    scores = df['niqe_score'].values

    # Calculate translation coefficients for the scores
    scale, shift = calculate_translation_coefficients(scores)

    # Translate the scores
    df['translated_brisque_score'] = translate_scores(scores, scale, shift)

    # Calculate additional score using transform_score function
    df['transformed_mos'] = df['translated_brisque_score'].apply(transform_score)

    # Write to new CSV file
    df.to_csv(output_csv_path, index=False)

    return df  # Returning the modified dataframe


# Example usage
csv_path = 'analyze_niqe/niqe_scores_LIVE2.csv'
output_csv_path = 'analyze_niqe/output_scores_LIVE2.csv'

# Plotting the distribution of the original NIQE scores
df = pd.read_csv(csv_path)
plt.figure(figsize=(10, 6))
sns.histplot(df['niqe_score'], bins=30, kde=True)
plt.title('Distribution of NIQE Scores')
plt.xlabel('NIQE Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Process the scores and read the output for plotting
df_transformed = process_scores(csv_path, output_csv_path)
df_output = pd.read_csv(output_csv_path)

# Plotting the NIQE to BRISQUE Score Translation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='niqe_score', y='translated_brisque_score', data=df_output)
plt.title('NIQE to BRISQUE Score Translation')
plt.xlabel('Original NIQE Score')
plt.ylabel('Translated BRISQUE Score')
plt.grid(True)
plt.show()
