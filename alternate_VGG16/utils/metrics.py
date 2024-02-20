import tensorflow as tf
from keras import backend as K
from scipy.stats import spearmanr


# K.mean(...) calculează media acestor valori, ceea ce reprezintă acuratețea
def custom_accuracy(y_true, y_pred, threshold=0.1):
    return K.mean(K.cast(K.abs(y_true - y_pred) < threshold, 'float32'))


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def srocc(y_true, y_pred):
    def compute_srocc(y_true, y_pred):
        srocc_value, _ = spearmanr(y_true, y_pred)
        return srocc_value

    srocc_value = tf.py_function(compute_srocc, [y_true, y_pred], tf.float32)
    return srocc_value


def plcc(y_true, y_pred):
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred
    numerator = K.sum(centered_true * centered_pred)
    denominator = K.sqrt(K.sum(K.square(centered_true)) * K.sum(K.square(centered_pred)))
    plcc_value = numerator / denominator
    return plcc_value
