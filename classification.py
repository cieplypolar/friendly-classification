import numpy as np
from abc import ABC, abstractmethod
from gradient import BaseGradient
from activation import sigmoid

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def squared_mean_error(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def zero_one_error(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y != y_pred)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    '''
    Hardcoded values for y {2, 4}
    '''
    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        true_positives = np.sum((y == 4) & (y_pred == 4))
        predicted_positives = np.sum(y_pred == 4)
        return true_positives / predicted_positives if predicted_positives > 0 else np.nan

    def sensitivity(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        true_positives = np.sum((y == 4) & (y_pred == 4))
        actual_positives = np.sum(y == 4)
        return true_positives / actual_positives if actual_positives > 0 else np.nan


class LogisticRegression(BaseModel):
    def __init__(
            self,
            grad: BaseGradient,
            regularization: float = 0,
            standardize: bool = True,
            fit_intercept: bool = True,
            alpha: float = 0.01,
            epochs: int = 100,
            eps: float = 0.1):
        self.intercept = None
        self.theta = None
        self.means = None
        self.stds = None
        self.grad = grad
        self.regularization = regularization
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.epochs = epochs
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        m, d = X.shape
        y_transformed = (y.copy() - 2) // 2
        self.theta = np.zeros(d).reshape(-1, 1)
        self.intercept = 0
        if (self.standardize):
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
        else:
            self.means = np.zeros(d)
            self.stds = np.ones(d)
        X_scaled = (X.copy() - self.means) / self.stds

        for _ in range(self.epochs):
            z = self.intercept + X_scaled @ self.theta
            y_pred = sigmoid(z)

            dintercept, dtheta = self.grad.calculate_gradient(
                self.regularization, X_scaled, self.theta, self.fit_intercept, y_pred, y_transformed
            )
            if not self.fit_intercept:
                dintercept = 0
            self.intercept -= (self.alpha) * dintercept
            self.theta -= (self.alpha / m) * dtheta
            if np.linalg.norm(dtheta) < self.eps:
                break

    '''
    Hardcoded values for y {2, 4}
    '''
    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.intercept + X @ self.theta
        y_pred = sigmoid(z)
        y_pred = np.where(y_pred > 0.5, 4, 2)
        return y_pred.reshape(-1, 1)

    def reset(self) -> None:
        self.intercept = None
        self.theta = None
        self.means = None
        self.stds = None

    def binary_cross_entropy(
            self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = sigmoid(X @ self.theta + self.intercept)
        y_transformed = (y.copy() - 2) // 2
        return -np.mean(y_transformed * np.log(y_pred) + (1 - y_transformed) * np.log(1 - y_pred))

class NaiveBayes(BaseModel):
    """
    We have strong assumptions about the data:
    1. x_j in {1, ..., 10}
    2. y in {2, 4}
    I can abuse this fact and hardcode some arrays ;)
    """

    def __init__(self, laplace: bool = True, boundary: float = 0.5):
        self.class_mle = np.zeros(2)
        self.feature_mle = [None] * 2
        self.laplace = int(laplace)
        self.boundary = boundary

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_transformed = (y.copy() - 2) // 2
        m, _ = X.shape
        n_y1 = np.sum(y_transformed)
        n_y0 = m - n_y1
        self.class_mle[0] = (1 * self.laplace + n_y0) / (m + 2 * self.laplace)
        self.class_mle[1] = (1 * self.laplace + n_y1) / (m + 2 * self.laplace)

        X_0 = X[y_transformed.flatten() == 0]
        feature_count = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=11)[1:], axis=0, arr=X_0
        ).T
        self.feature_mle[0] = (feature_count + (1 * self.laplace)) / (
            10 * self.laplace + n_y0
        )

        X_1 = X[y_transformed.flatten() == 1]
        feature_count = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=11)[1:], axis=0, arr=X_1
        ).T
        self.feature_mle[1] = (feature_count + (1 * self.laplace)) / (
            10 * self.laplace + n_y1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        2 nested for loops will be slow, but it's ok for now
        """
        y_pred = []
        for x in X:
            p_y1 = self.class_mle[1]
            p_y0 = self.class_mle[0]
            for i, feature in enumerate(x):
                p_y1 *= self.feature_mle[1][i][feature - 1]
                p_y0 *= self.feature_mle[0][i][feature - 1]
            p_y = p_y1 / (p_y1 + p_y0)
            y_pred.append(4 if p_y > self.boundary else 2)

        return np.array(y_pred).reshape(-1, 1)

    def reset(self) -> None:
        self.class_mle = np.zeros(2)
        self.feature_mle = [None] * 2
