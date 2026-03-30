import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from neural_network import CNN


def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X = digits.images.astype(np.float64) / 16.0
    X = X[:, np.newaxis, :, :]
    y = digits.target.astype(np.int64)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model():
    model = CNN()

    model.add_layer(Conv2D(in_channels=1, out_channels=6, kernel_size=3, padding=1))
    model.add_layer(ReLU())
    model.add_layer(MaxPool2D(pool_size=2, stride=2))

    model.add_layer(Flatten())
    model.add_layer(Dense(in_features=6 * 4 * 4, out_features=32))
    model.add_layer(ReLU())
    model.add_layer(Dense(in_features=32, out_features=10))

    model.compile(optimizer="adam", learning_rate=0.001)
    return model


def main():
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_data()

    model = build_model()
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=5,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
