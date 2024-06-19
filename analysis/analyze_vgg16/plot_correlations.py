import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_correlations(predicted_csv, ground_truth_csv):
    # Load the CSV files into Pandas DataFrames
    predicted_df = pd.read_csv(predicted_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    # Merge the DataFrames on the 'image_name' column to ensure matching scores are compared
    merged_df = pd.merge(predicted_df, ground_truth_df, on='image_name', how='inner')

    # Extract the MOS scores after merging
    predicted_scores = merged_df['MOS_predicted_score'].values
    ground_truth_scores = merged_df['MOS'].values

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv('merged_scores.csv', index=False)

    # Calculate Spearman's Rank Correlation Coefficient and Pearson's Linear Correlation Coefficient
    srcc, _ = spearmanr(predicted_scores, ground_truth_scores)
    plcc, _ = pearsonr(predicted_scores, ground_truth_scores)
    # srcc = merged_df[['MOS_predicted_score', 'MOS']].corr(method='spearman').iloc[0, 1]
    # plcc = merged_df[['MOS_predicted_score', 'MOS']].corr(method='pearson').iloc[0, 1]

    return srcc, plcc


def plot_correlations(srcc, plcc):
    correlations = [plcc, srcc]
    methods = ['PLCC', 'SRCC']

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.barplot(x=methods, y=correlations, palette='coolwarm')
    plt.title('Correlation to Ground Truth MOS')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Correlation Type')
    plt.ylim(0, 1)  # Assuming correlation values will be positive; adjust as necessary
    for index, value in enumerate(correlations):
        plt.text(index, value + 0.02, f"{value:.4f}", ha='center')

    plt.show()


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def plot_error_metrics(rmse, mae):
    metrics = [rmse, mae]
    labels = ['RMSE', 'MAE']

    # Plotting
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=metrics, palette='viridis')
    plt.title('Error Metrics for Predicted Scores')
    plt.ylabel('Value')
    for index, value in enumerate(metrics):
        plt.text(index, value + 0.01, f"{value:.4f}", ha='center')

    plt.ylim(0, max(metrics) + 0.1)  # Adjust as necessary
    plt.show()


def main():
    predicted_csv = 'predicted_scores_on_LIVE2_model_trained_on_Koniq10k_130.csv'
    # ground_truth_csv = '../../alternate_VGG16/data/Koniq_10k/koniq10k_scores_and_distributions.csv'
    ground_truth_csv = '../../alternate_VGG16/LIVE_release2/LIVE2_MOS_scores.csv'

    # Calculate correlations
    srcc, plcc = calculate_correlations(predicted_csv, ground_truth_csv)

    # Print the correlation coefficients
    print(f"Spearman Rank Correlation Coefficient (SRCC): {srcc:.4f}")
    print(f"Pearson Linear Correlation Coefficient (PLCC): {plcc:.4f}")

    # Plot the correlations
    plot_correlations(srcc, plcc)

    # Calculating RMSE and MAE
    merged_df = pd.merge(pd.read_csv(predicted_csv), pd.read_csv(ground_truth_csv), on='image_name', how='inner')
    rmse = calculate_rmse(merged_df['MOS'].values, merged_df['MOS_predicted_score'].values)
    mae = calculate_mae(merged_df['MOS'].values, merged_df['MOS_predicted_score'].values)

    # Print RMSE and MAE
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot RMSE and MAE
    plot_error_metrics(rmse, mae)


if __name__ == "__main__":
    main()
