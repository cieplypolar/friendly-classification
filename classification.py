import numpy as np
from abc import ABC, abstractmethod


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


class LogisticRegression(BaseModel):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class NaiveBayes(BaseModel):
    """
    We have strong assumptions about the data:
    1. x_j in {1, ..., 10}
    2. y in {2, 4}
    I can abuse this fact and hardcode some arrays ;)
    """

    def __init__(self, laplace: bool = True):
        self.class_mle = np.zeros(2)
        self.feature_mle = [None] * 2
        self.laplace = int(laplace)

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
            y_pred.append(4 if p_y > 0.5 else 2)

        return np.array(y_pred).reshape(-1, 1)

    def reset(self) -> None:
        self.class_mle = np.zeros(2)
        self.feature_mle = [None] * 2
        self.laplace = 1
