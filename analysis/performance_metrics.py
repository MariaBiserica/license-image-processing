import numpy as np


def srcc(predicted, ground_truth):
    """
    Calculate the Spearman Rank-Order Correlation Coefficient (SRCC).

    SRCC is a nonparametric measure of rank correlation that assesses how well the relationship
    between two variables can be described using a monotonic function.

    :param predicted: An array of predicted quality scores.
    :param ground_truth: An array of subjectively assessed ground truth quality scores.
    :return: The SRCC value.
    """
    n = len(predicted)
    rank_predicted = np.argsort(np.argsort(predicted))
    rank_ground_truth = np.argsort(np.argsort(ground_truth))
    d = rank_predicted - rank_ground_truth
    return 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))


def plcc(predicted, ground_truth):
    """
    Calculate Pearson’s Linear Correlation Coefficient (PLCC).

    PLCC measures the linear correlation between two variables. It has a value between +1 and -1,
    where 1 is total positive linear correlation, 0 is no linear correlation, and -1 is total
    negative linear correlation.

    :param predicted: An array of predicted quality scores.
    :param ground_truth: An array of subjectively assessed ground truth quality scores.
    :return: The PLCC value.
    """
    predicted_mean = np.mean(predicted)
    ground_truth_mean = np.mean(ground_truth)
    numerator = np.sum((predicted - predicted_mean) * (ground_truth - ground_truth_mean))
    denominator = np.sqrt(np.sum((predicted - predicted_mean)**2) * np.sum((ground_truth - ground_truth_mean)**2))
    return numerator / denominator


def rmse(predicted, ground_truth):
    """
    Calculate the Root Mean Squared Error (RMSE).

    RMSE is a frequently used measure of the differences between values predicted by a model
    or an estimator and the values observed.

    :param predicted: An array of predicted quality scores.
    :param ground_truth: An array of subjectively assessed ground truth quality scores.
    :return: The RMSE value.
    """
    return np.sqrt(np.mean((predicted - ground_truth)**2))


def mae(predicted, ground_truth):
    """
    Calculate the Mean Absolute Error (MAE).

    MAE is a measure of errors between paired observations expressing the same phenomenon.

    :param predicted: An array of predicted quality scores.
    :param ground_truth: An array of subjectively assessed ground truth quality scores.
    :return: The MAE value.
    """
    return np.mean(np.abs(predicted - ground_truth))
