import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden_units=1000, alpha=1.0):
        self.n_hidden_units = n_hidden_units
        self.alpha = alpha
        self.normalizer = Normalizer()
        self.ridge = Ridge(alpha=self.alpha)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.normalizer.fit(X)
        X = self.normalizer.transform(X)

        # Generate random weights and biases for hidden layer
        self.coef_hidden_ = np.random.normal(loc=0.0, scale=1.0, size=(X.shape[1], self.n_hidden_units))
        self.intercept_hidden_ = np.random.uniform(low=-1.0, high=1.0, size=self.n_hidden_units)

        # Compute hidden layer activations
        G = self._sigmoid(np.dot(X, self.coef_hidden_) + self.intercept_hidden_)

        # Train output weights
        self.ridge.fit(G, y)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self.normalizer.transform(X)

        # Compute hidden layer activations
        G = self._sigmoid(np.dot(X, self.coef_hidden_) + self.intercept_hidden_)

        # Compute output layer
        predictions = self.ridge.predict(G)
        return predictions
