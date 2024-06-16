import pandas as pd
import matplotlib.pyplot as plt


# Function to read the CSV file and generate the distribution chart
def generate_score_distribution_chart(csv_file_name):
    # Reading the CSV file
    data = pd.read_csv(csv_file_name)

    # Checking if the 'scores' column exists in the data
    if 'Sharpness_Score' not in data.columns:
        raise ValueError("The CSV file does not contain a column named 'scores'.")

    # Generating the distribution chart
    plt.figure(figsize=(10, 6))
    plt.hist(data['Sharpness_Score'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Score Distribution')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Example usage
csv_file_name = 'Koniq10k_sharpness_scores.csv'
generate_score_distribution_chart(csv_file_name)
