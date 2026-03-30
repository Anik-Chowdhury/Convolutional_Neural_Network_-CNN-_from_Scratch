import numpy as np

from activation_functions import ActivationFunctions


def _get_im2col_indices(x_shape, field_height, field_width, padding, stride):
    _, channels, height, width = x_shape

    out_height = (height + 2 * padding - field_height) // stride + 1
    out_width = (width + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height)
    j0 = np.tile(j0, channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), field_height * field_width).reshape(-1, 1)

    return k, i, j


def _im2col_indices(x, field_height, field_width, padding, stride):
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    k, i, j = _get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * x.shape[1], -1)
    return cols


def _col2im_indices(cols, x_shape, field_height, field_width, padding, stride):
    batch_size, channels, height, width = x_shape
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding

    x_padded = np.zeros((batch_size, channels, padded_height, padded_width), dtype=cols.dtype)
    k, i, j = _get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(channels * field_height * field_width, -1, batch_size)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D:
    """A simple 2D convolution layer (NCHW format)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * self.kernel_size * self.kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, self.kernel_size, self.kernel_size) * scale
        self.biases = np.zeros(out_channels)

        self.input_shape = None
        self.x_cols = None
        self.out_h = None
        self.out_w = None

        self.dweights = None
        self.dbiases = None

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape

        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {in_channels}.")

        self.input_shape = x.shape
        self.out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.x_cols = _im2col_indices(
            x,
            field_height=self.kernel_size,
            field_width=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        w_cols = self.weights.reshape(self.out_channels, -1)
        out = w_cols @ self.x_cols + self.biases.reshape(-1, 1)
        out = out.reshape(self.out_channels, self.out_h, self.out_w, batch_size).transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        batch_size = self.input_shape[0]

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.dbiases = np.sum(dout_reshaped, axis=1) / batch_size
        self.dweights = (dout_reshaped @ self.x_cols.T).reshape(self.weights.shape) / batch_size

        w_cols = self.weights.reshape(self.out_channels, -1)
        dx_cols = w_cols.T @ dout_reshaped
        dx = _col2im_indices(
            dx_cols,
            x_shape=self.input_shape,
            field_height=self.kernel_size,
            field_width=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        return dx


class ReLU:
    """ReLU activation layer."""

    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return ActivationFunctions.relu(x)

    def backward(self, dout):
        return dout * ActivationFunctions.derivative_of_relu(self.input)


class MaxPool2D:
    """2D max pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.input_shape = None
        self.x_cols = None
        self.max_indices = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        batch_size, channels, in_h, in_w = x.shape
        self.input_shape = x.shape

        self.out_h = (in_h - self.pool_size) // self.stride + 1
        self.out_w = (in_w - self.pool_size) // self.stride + 1

        x_reshaped = x.reshape(batch_size * channels, 1, in_h, in_w)
        self.x_cols = _im2col_indices(
            x_reshaped,
            field_height=self.pool_size,
            field_width=self.pool_size,
            padding=0,
            stride=self.stride,
        )

        self.max_indices = np.argmax(self.x_cols, axis=0)
        out = self.x_cols[self.max_indices, np.arange(self.max_indices.size)]
        out = out.reshape(self.out_h, self.out_w, batch_size, channels).transpose(2, 3, 0, 1)

        return out

    def backward(self, dout):
        batch_size, channels, in_h, in_w = self.input_shape

        dout_flat = dout.transpose(2, 3, 0, 1).reshape(-1)
        dx_cols = np.zeros_like(self.x_cols)
        dx_cols[self.max_indices, np.arange(self.max_indices.size)] = dout_flat

        dx = _col2im_indices(
            dx_cols,
            x_shape=(batch_size * channels, 1, in_h, in_w),
            field_height=self.pool_size,
            field_width=self.pool_size,
            padding=0,
            stride=self.stride,
        )

        return dx.reshape(self.input_shape)


class Flatten:
    """Flatten layer from NCHW to (N, features)."""

    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)


class Dense:
    """Fully connected layer."""

    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.biases = np.zeros(out_features)

        self.input = None
        self.dweights = None
        self.dbiases = None

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, dout):
        batch_size = self.input.shape[0]
        self.dweights = (self.input.T @ dout) / batch_size
        self.dbiases = np.sum(dout, axis=0) / batch_size
        return dout @ self.weights.T
