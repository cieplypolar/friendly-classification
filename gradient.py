import numpy as np
from abc import ABC, abstractmethod

class BaseGradient(ABC):
    @abstractmethod
    def calculate_gradient(
        self,
        regularization: float,
        X: np.ndarray,
        theta: np.ndarray,
        intercept: np.float64,
        y_pred: np.ndarray,
        y: np.ndarray) -> tuple[np.float64, np.ndarray]:
        pass

class GradientDescent(BaseGradient):
    def calculate_gradient(
        self,
        regularization: float,
        X: np.ndarray,
        theta: np.ndarray,
        intercept: np.float64,
        y_pred: np.ndarray,
        y: np.ndarray) -> tuple[np.float64, np.ndarray]:

        regularization_term = regularization * (2 * theta)
        error = y_pred - y
        gradient = X.T @ error + regularization_term
        bias_gradient = np.sum(error)
        return bias_gradient, gradient


