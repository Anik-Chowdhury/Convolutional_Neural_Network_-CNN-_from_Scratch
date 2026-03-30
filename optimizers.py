import numpy as np


class Optimizer:
    """Base class for optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def initialize_parameters(self, layer):
        pass

    def update_layer(self, layer):
        raise NotImplementedError("Subclasses must implement update_layer().")


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and Nesterov."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov

    def initialize_parameters(self, layer):
        if self.momentum > 0.0:
            layer.Vdw = np.zeros_like(layer.weights)
            layer.Vdb = np.zeros_like(layer.biases)

    def update_layer(self, layer):
        if self.momentum > 0.0:
            if not hasattr(layer, "Vdw") or not hasattr(layer, "Vdb"):
                self.initialize_parameters(layer)

            layer.Vdw = self.momentum * layer.Vdw - self.learning_rate * layer.dweights
            layer.Vdb = self.momentum * layer.Vdb - self.learning_rate * layer.dbiases

            if self.nesterov:
                layer.weights += self.momentum * layer.Vdw - self.learning_rate * layer.dweights
                layer.biases += self.momentum * layer.Vdb - self.learning_rate * layer.dbiases
            else:
                layer.weights += layer.Vdw
                layer.biases += layer.Vdb
        else:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize_parameters(self, layer):
        layer.m_dw = np.zeros_like(layer.weights)
        layer.v_dw = np.zeros_like(layer.weights)
        layer.m_db = np.zeros_like(layer.biases)
        layer.v_db = np.zeros_like(layer.biases)
        layer.t = 0

    def update_layer(self, layer):
        if not hasattr(layer, "m_dw"):
            self.initialize_parameters(layer)

        layer.t += 1

        layer.m_dw = self.beta1 * layer.m_dw + (1.0 - self.beta1) * layer.dweights
        layer.v_dw = self.beta2 * layer.v_dw + (1.0 - self.beta2) * (layer.dweights ** 2)

        layer.m_db = self.beta1 * layer.m_db + (1.0 - self.beta1) * layer.dbiases
        layer.v_db = self.beta2 * layer.v_db + (1.0 - self.beta2) * (layer.dbiases ** 2)

        m_dw_hat = layer.m_dw / (1.0 - self.beta1 ** layer.t)
        v_dw_hat = layer.v_dw / (1.0 - self.beta2 ** layer.t)

        m_db_hat = layer.m_db / (1.0 - self.beta1 ** layer.t)
        v_db_hat = layer.v_db / (1.0 - self.beta2 ** layer.t)

        layer.weights -= self.learning_rate * m_dw_hat / (np.sqrt(v_dw_hat) + self.epsilon)
        layer.biases -= self.learning_rate * m_db_hat / (np.sqrt(v_db_hat) + self.epsilon)


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon

    def initialize_parameters(self, layer):
        layer.cache_dw = np.zeros_like(layer.weights)
        layer.cache_db = np.zeros_like(layer.biases)

    def update_layer(self, layer):
        if not hasattr(layer, "cache_dw") or not hasattr(layer, "cache_db"):
            self.initialize_parameters(layer)

        layer.cache_dw = self.rho * layer.cache_dw + (1.0 - self.rho) * (layer.dweights ** 2)
        layer.cache_db = self.rho * layer.cache_db + (1.0 - self.rho) * (layer.dbiases ** 2)

        layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(layer.cache_dw) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(layer.cache_db) + self.epsilon)
