import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


def prepare_and_combine_data(brisque_csv_path, niqe_csv_path, gt_csv_path, output_csv_path):
    # Reading the CSV files
    brisque_df = pd.read_csv(brisque_csv_path)
    niqe_df = pd.read_csv(niqe_csv_path)
    gt_df = pd.read_csv(gt_csv_path)

    # Rename columns as per the new requirement
    brisque_df.rename(columns={'brisque_score': 'brisque', 'MOS_zscore_transformed_to_brisque_scale': 'MOS_zscore_transformed_to_brisque_scale'}, inplace=True)
    niqe_df.rename(columns={'translated_brisque_score': 'niqe_transformed_to_brisque_scale'}, inplace=True)
    gt_df.rename(columns={'MOS': 'MOS_zscore_transformed_to_brisque_scale'}, inplace=True)

    # Merge BRISQUE and NIQE data based on the image name with an 'inner' join
    combined_df = pd.merge(brisque_df[['image_name', 'brisque', 'MOS_zscore_transformed_to_brisque_scale']],
                           niqe_df[['image_name', 'niqe_transformed_to_brisque_scale']],
                           on='image_name',
                           how='inner')

    # Round the scores to four decimal places
    combined_df['brisque'] = combined_df['brisque'].round(4)
    combined_df['niqe_transformed_to_brisque_scale'] = combined_df['niqe_transformed_to_brisque_scale'].round(4)
    combined_df['MOS_zscore_transformed_to_brisque_scale'] = combined_df['MOS_zscore_transformed_to_brisque_scale'].round(4)

    # Saving the combined data into a new CSV file
    combined_df.to_csv(output_csv_path, index=False)

    print(f'The combined file has been saved as: {output_csv_path}')

    return combined_df


def plot_correlations(df):
    # Update column names
    brisque_corr = df[['brisque', 'MOS_zscore_transformed_to_brisque_scale']].corr(method='pearson').iloc[0, 1]
    niqe_corr = df[['niqe_transformed_to_brisque_scale', 'MOS_zscore_transformed_to_brisque_scale']].corr(method='pearson').iloc[0, 1]

    correlations = [brisque_corr, niqe_corr]
    methods = ['BRISQUE', 'NIQE Transformed to BRISQUE Scale']

    plt.figure(figsize=(8, 6))
    sns.barplot(x=methods, y=correlations, palette='coolwarm')
    plt.title('Pearson Correlation to MOS (Z-Score Transformed to BRISQUE Scale)')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.xlabel('Method')
    plt.ylim(0, 1)
    for index, value in enumerate(correlations):
        plt.text(index, value + 0.02, f"{value:.4f}", ha='center')
    plt.show()


def plot_regression_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.regplot(ax=axes[0], x='brisque', y='MOS_zscore_transformed_to_brisque_scale', data=df,
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_title('BRISQUE vs MOS (Z-Score Transformed)')
    axes[0].set_xlabel('BRISQUE Score')
    axes[0].set_ylabel('MOS (Z-Score Transformed)')

    sns.regplot(ax=axes[1], x='niqe_transformed_to_brisque_scale', y='MOS_zscore_transformed_to_brisque_scale', data=df,
                scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
    axes[1].set_title('NIQE (Transformed to BRISQUE Scale) vs MOS (Z-Score Transformed)')
    axes[1].set_xlabel('NIQE (Transformed to BRISQUE Scale)')

    plt.tight_layout()
    plt.show()


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def plot_error_metrics(df):
    brisque_rmse = calculate_rmse(df['MOS_zscore_transformed_to_brisque_scale'], df['brisque'])
    niqe_rmse = calculate_rmse(df['MOS_zscore_transformed_to_brisque_scale'], df['niqe_transformed_to_brisque_scale'])
    brisque_mae = calculate_mae(df['MOS_zscore_transformed_to_brisque_scale'], df['brisque'])
    niqe_mae = calculate_mae(df['MOS_zscore_transformed_to_brisque_scale'], df['niqe_transformed_to_brisque_scale'])

    rmses = [brisque_rmse, niqe_rmse]
    maes = [brisque_mae, niqe_mae]
    methods = ['BRISQUE', 'NIQE Transformed to BRISQUE Scale']

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(ax=ax[0], x=methods, y=rmses, palette='pastel')
    ax[0].set_title('RMSE for BRISQUE and NIQE (Transformed)')
    ax[0].set_ylabel('RMSE')
    for index, value in enumerate(rmses):
        ax[0].text(index, value + 0.01, f"{value:.4f}", ha='center')

    sns.barplot(ax=ax[1], x=methods, y=maes, palette='pastel')
    ax[1].set_title('MAE for BRISQUE and NIQE (Transformed)')
    ax[1].set_ylabel('MAE')
    for index, value in enumerate(maes):
        ax[1].text(index, value + 0.01, f"{value:.4f}", ha='center')

    plt.tight_layout()
    plt.show()


def main():
    brisque_csv_path = 'analyze_brisque/output_scores_Koniq10k_brisque_original_scale.csv'
    niqe_csv_path = 'analyze_niqe/output_scores_Koniq10k.csv'
    gt_csv_path = '../VGG16/data/koniq10k_scores_and_distributions.csv'
    output_csv_path = 'brisque_niqe_mos_scores_Koniq10k_original_scale.csv'

    combined_df = prepare_and_combine_data(brisque_csv_path, niqe_csv_path, gt_csv_path, output_csv_path)
    plot_correlations(combined_df)
    plot_regression_scatter(combined_df)
    plot_error_metrics(combined_df)


if __name__ == "__main__":
    main()
