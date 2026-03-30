import numpy as np


class ActivationFunctions:
    """Common activation functions used in the CNN."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def derivative_of_relu(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(x.dtype)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
