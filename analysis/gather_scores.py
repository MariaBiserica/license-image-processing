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

    # Rename the column for the transformed scores to avoid confusion
    brisque_df.rename(columns={'transformed_mos': 'brisque_transformed_mos'}, inplace=True)
    niqe_df.rename(columns={'transformed_mos': 'niqe_transformed_mos'}, inplace=True)

    # First, merge BRISQUE and NIQE data based on the image name with an 'inner' join
    combined_df = pd.merge(brisque_df[['image_name', 'brisque_transformed_mos']],
                           niqe_df[['image_name', 'niqe_transformed_mos']],
                           on='image_name',
                           how='inner')

    # Then, merge the combined BRISQUE and NIQE dataframe with the GT scores, also using an 'inner' join
    # This ensures only images present in all three datasets are included in the final dataframe
    combined_df = pd.merge(combined_df, gt_df[['image_name', 'MOS']],
                           on='image_name',
                           how='inner')

    # Round the scores to four decimal places
    combined_df['brisque_transformed_mos'] = combined_df['brisque_transformed_mos'].round(4)
    combined_df['niqe_transformed_mos'] = combined_df['niqe_transformed_mos'].round(4)
    combined_df['MOS'] = combined_df['MOS'].round(4)

    # Saving the combined data into a new CSV file
    combined_df.to_csv(output_csv_path, index=False)

    print(f'The combined file has been saved as: {output_csv_path}')

    return combined_df


def plot_correlations(df):
    """
    Plot Pearson correlation coefficients for BRISQUE and NIQE methods against ground truth scores.
    PCC measures the linear correlation between two sets of continuous data, indicating the strength and direction
    of a linear relationship between them. It evaluates:

    1.Direction of the Relationship: A positive PCC indicates that as one variable increases, the other variable also
    increases. In the context of image quality assessment, a positive PCC would suggest that as the quality score
    increases (improves), the perceived quality (MOS) also increases, which is the expected behavior.

    2.Strength of the Relationship: The magnitude of the PCC (ignoring the sign) indicates the strength of the linear
    relationship, ranging from 0 to 1. A PCC close to 1 suggests a strong linear relationship, meaning the model's
    scores align well with human perception (MOS) in a linear manner. A low PCC indicates a weak linear relationship.

    Y-Axis (Pearson Correlation Coefficient): The values range from 0 to 1, which represent the strength of the linear
    relationship between the predicted quality scores (from BRISQUE and NIQE) and the ground truth MOS. A value of 1
    would indicate a perfect positive linear correlation, and a value closer to 0 would indicate a weaker correlation.

    X-Axis (Method): There are two bars, each corresponding to one of the methods (BRISQUE on the left and NIQE on the
    right).

    Height of the Bars: The height of each bar indicates the PCC value for that method. A taller bar would indicate a
    stronger correlation with the ground truth MOS.
    """
    # Calculate Pearson correlation coefficients
    brisque_corr = df[['brisque_transformed_mos', 'MOS']].corr(method='pearson').iloc[0, 1]
    niqe_corr = df[['niqe_transformed_mos', 'MOS']].corr(method='pearson').iloc[0, 1]

    # Prepare data for plotting
    correlations = [brisque_corr, niqe_corr]
    methods = ['BRISQUE', 'NIQE']

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.barplot(x=methods, y=correlations, palette='coolwarm')
    plt.title('Pearson Correlation to Ground Truth MOS')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.xlabel('Method')
    plt.ylim(0, 1)  # Assuming correlation values will be positive; adjust as necessary
    for index, value in enumerate(correlations):
        plt.text(index, value + 0.02, f"{value:.4f}", ha='center')

    plt.show()


def plot_regression_scatter(df):
    """
    Plot scatter plots with regression lines for BRISQUE and NIQE against the MOS scores.
    This visualization helps to understand the degree to which the quality scores from each method
    predict the ground truth MOS.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Scatter plot for BRISQUE
    sns.regplot(ax=axes[0], x='brisque_transformed_mos', y='MOS', data=df,
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_title('BRISQUE vs MOS')
    axes[0].set_xlabel('BRISQUE Transformed MOS')
    axes[0].set_ylabel('Ground Truth MOS')

    # Scatter plot for NIQE
    sns.regplot(ax=axes[1], x='niqe_transformed_mos', y='MOS', data=df,
                scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
    axes[1].set_title('NIQE vs MOS')
    axes[1].set_xlabel('NIQE Transformed MOS')
    # The Y label is shared, set only on the left plot

    plt.tight_layout()
    plt.show()


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def plot_error_metrics(df):
    """
    Plot RMSE and MAE for BRISQUE and NIQE methods against ground truth scores.
    RMSE and MAE provide different perspectives on the prediction errors:
    - RMSE gives a relatively high weight to large errors.
    - MAE treats all errors equally.
    """
    # Calculate error metrics
    brisque_rmse = calculate_rmse(df['MOS'], df['brisque_transformed_mos'])
    niqe_rmse = calculate_rmse(df['MOS'], df['niqe_transformed_mos'])
    brisque_mae = calculate_mae(df['MOS'], df['brisque_transformed_mos'])
    niqe_mae = calculate_mae(df['MOS'], df['niqe_transformed_mos'])

    rmses = [brisque_rmse, niqe_rmse]
    maes = [brisque_mae, niqe_mae]
    methods = ['BRISQUE', 'NIQE']

    # Plotting RMSE and MAE
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE Plot
    sns.barplot(ax=ax[0], x=methods, y=rmses, palette='pastel')
    ax[0].set_title('RMSE for BRISQUE and NIQE')
    ax[0].set_ylabel('RMSE')
    for index, value in enumerate(rmses):
        ax[0].text(index, value + 0.01, f"{value:.4f}", ha='center')

    # MAE Plot
    sns.barplot(ax=ax[1], x=methods, y=maes, palette='pastel')
    ax[1].set_title('MAE for BRISQUE and NIQE')
    ax[1].set_ylabel('MAE')
    for index, value in enumerate(maes):
        ax[1].text(index, value + 0.01, f"{value:.4f}", ha='center')

    plt.tight_layout()
    plt.show()


def main():
    # Paths to the existing CSV files
    brisque_csv_path = 'analyze_brisque/output_scores_Koniq10k_brisque_original_scale.csv'
    niqe_csv_path = 'analyze_niqe/output_scores_Koniq10k.csv'
    gt_csv_path = '../VGG16/data/koniq10k_scores_and_distributions.csv'
    output_csv_path = 'brisque_niqe_mos_scores_Koniq10k_original_scale.csv'

    # Process and combine the data
    combined_df = prepare_and_combine_data(brisque_csv_path, niqe_csv_path, gt_csv_path, output_csv_path)

    # Plot the correlations
    plot_correlations(combined_df)

    # Plot regression scatter plots
    plot_regression_scatter(combined_df)

    # Plot error metrics
    plot_error_metrics(combined_df)


if __name__ == "__main__":
    main()
