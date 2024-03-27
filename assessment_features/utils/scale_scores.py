import pandas as pd


def scale_scores_in_csv(csv_path, output_path):
    """
    Reads a CSV file, scales the 'Cimage' score to the range 1-5, and saves a new CSV with the scaled scores.

    :param csv_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file with scaled scores.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract the 'Cimage' scores
    scores = df['Noise_Score']

    # Find the minimum and maximum score for scaling
    min_score, max_score = scores.min(), scores.max()

    # Define new range
    new_min, new_max = 1, 5

    # Scale the scores
    df['Noise_Score_to_MOS_scale'] = new_min + (new_max - new_min) * (scores - min_score) / (max_score - min_score)

    # Save the new DataFrame to a CSV
    df.to_csv(output_path, index=False)

    return df  # Return the DataFrame for verification if needed

