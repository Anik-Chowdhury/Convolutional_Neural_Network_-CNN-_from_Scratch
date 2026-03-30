# Convolutional Neural Network (CNN) from Scratch (NumPy)

This repository is a pure-NumPy implementation of a CNN with manual forward/backward propagation.
No deep learning frameworks are used.

## Highlights

- CNN layers implemented from scratch:
  - `Conv2D`
  - `MaxPool2D`
  - `ReLU`
  - `Flatten`
  - `Dense`
- Training pipeline from scratch:
  - Softmax + cross-entropy loss
  - Backpropagation through all layers
  - Mini-batch training loop
- Multiple optimizers:
  - `SGD` (with momentum/Nesterov support)
  - `RMSprop`
  - `Adam`
- Fast convolution/pooling internals using `im2col` / `col2im`

## Project Structure

- `activation_functions.py` - ReLU and Softmax
- `layers.py` - `Conv2D`, `MaxPool2D`, `ReLU`, `Flatten`, `Dense`
- `optimizers.py` - `SGD`, `RMSprop`, `Adam`
- `neural_network.py` - `CNN` class with `fit`, `evaluate`, `predict`
- `main.py` - default training run on `sklearn` digits dataset (8x8)
- `train_mnist10.py` - MNIST/10 run (10,000 train, 10,000 test)

## Installation

```bash
pip install numpy scikit-learn
```

## Quick Start

Run the default digits example:

```bash
python main.py
```

This trains for 20 epochs on `sklearn.datasets.load_digits` and prints train/validation metrics.

## MNIST/10 Experiment (10 Epochs)

Run:

```bash
python train_mnist10.py
```

Setup used in this script:

- Dataset: MNIST (`0-9`) from OpenML
- Train size: `10,000` samples
- Test size: `10,000` samples
- Epochs: `10`
- Batch size: `128`
- Optimizer: `Adam` (`learning_rate=0.001`)

## Reported Results

### Digits (`main.py`)

- Final Test Loss: `0.1721`
- Final Test Accuracy: `0.9528`

### MNIST/10 (`train_mnist10.py`, 10 epochs)

- Final Test Loss: `0.105662`
- Final Test Accuracy: `0.968600`

## Default Model Architectures

### `main.py` (8x8 digits)

1. `Conv2D(1 -> 6, kernel=3, padding=1)`
2. `ReLU`
3. `MaxPool2D(2x2, stride=2)`
4. `Flatten`
5. `Dense(96 -> 32)`
6. `ReLU`
7. `Dense(32 -> 10)`

### `train_mnist10.py` (28x28 MNIST)

1. `Conv2D(1 -> 6, kernel=3, padding=1)`
2. `ReLU`
3. `MaxPool2D(2x2, stride=2)`
4. `Flatten`
5. `Dense(1176 -> 64)`
6. `ReLU`
7. `Dense(64 -> 10)`

## Notes

- Input tensor format is **NCHW**: `(batch, channels, height, width)`.
- This project is educational and intentionally keeps core operations transparent.
- You can adjust architecture, learning rate, epochs, and batch size in `main.py` or `train_mnist10.py`.
