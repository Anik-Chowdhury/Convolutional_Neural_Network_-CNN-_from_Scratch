import numpy as np

from activation_functions import ActivationFunctions
from optimizers import SGD, Adam, RMSprop


class CNN:
    """Sequential CNN with manual forward/backward propagation."""

    def __init__(self):
        self.layers = []
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer="adam", learning_rate=0.001, **optimizer_kwargs):
        if optimizer == "sgd":
            self.optimizer = SGD(learning_rate=learning_rate, **optimizer_kwargs)
        elif optimizer == "adam":
            self.optimizer = Adam(learning_rate=learning_rate, **optimizer_kwargs)
        elif optimizer == "rmsprop":
            self.optimizer = RMSprop(learning_rate=learning_rate, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _cross_entropy(self, y_true, probs):
        batch_size = y_true.shape[0]
        clipped = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.log(clipped[np.arange(batch_size), y_true]))

    def _loss_and_gradient(self, logits, y_true):
        probs = ActivationFunctions.softmax(logits)
        loss = self._cross_entropy(y_true, probs)

        grad = probs.copy()
        grad[np.arange(y_true.shape[0]), y_true] -= 1.0
        return loss, grad

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_trainable_layers(self):
        for layer in self.layers:
            if hasattr(layer, "weights") and hasattr(layer, "dweights"):
                self.optimizer.update_layer(layer)

    def fit(
        self,
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=None,
        shuffle=True,
        verbose=1,
    ):
        if self.optimizer is None:
            raise RuntimeError("Call compile() before fit().")

        num_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_epoch = X_train[indices]
                y_epoch = y_train[indices]
            else:
                X_epoch = X_train
                y_epoch = y_train

            epoch_loss_sum = 0.0
            epoch_correct = 0
            seen = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                logits = self.forward(X_batch)
                loss, grad = self._loss_and_gradient(logits, y_batch)

                self.backward(grad)
                self._update_trainable_layers()

                batch_n = X_batch.shape[0]
                epoch_loss_sum += loss * batch_n
                predictions = np.argmax(logits, axis=1)
                epoch_correct += int(np.sum(predictions == y_batch))
                seen += batch_n

            train_loss = epoch_loss_sum / seen
            train_acc = epoch_correct / seen
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)

            val_log = ""
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_acc)
                val_log = f" | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"

            if verbose and (epoch == 1 or epoch == epochs or epoch % verbose == 0):
                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"loss: {train_loss:.4f} | acc: {train_acc:.4f}{val_log}"
                )

    def predict_proba(self, X):
        logits = self.forward(X)
        return ActivationFunctions.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        probs = self.predict_proba(X)
        loss = self._cross_entropy(y, probs)
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == y)
        return loss, accuracy
