import time
from datetime import datetime

import numpy as np

np.random.seed(50000)


class Conv():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernels = []
        self.b = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)
        min = -0.1
        uniform_range = 0.1
        for i in range(self.out_channels):
            self.kernels.append(
                np.random.uniform(-uniform_range, uniform_range, ([self.in_channels, kernel_size, kernel_size])))
        self.kernels = np.array(self.kernels)
        self.cache = 0
        self.grads = 0
        self.padding = padding

    def forward(self, X):
        # Performs a forward convolution.

        n_C_prev, n_H_prev, n_W_prev = X.shape
        # X = self.zero_pad(X, pad=self.padding)

        n_C = self.out_channels
        n_H = int((n_H_prev + 2 * self.padding - self.kernel_size) / self.stride) + 1
        n_W = int((n_W_prev + 2 * self.padding - self.kernel_size) / self.stride) + 1
        arr = []
        arr.append(X)
        arr = np.array(arr)
        X_col = self.im2col(arr, self.kernel_size, self.kernel_size, self.stride, self.padding)
        w_col = self.kernels.reshape((self.out_channels, -1))
        b_col = self.b.reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, 1)).reshape((1, n_C, n_H, n_W))
        self.cache = arr, X_col, w_col
        return out[0]

    def im2col(self, X, HF, WF, stride, pad):
        # Transforms our input image into a matrix.
        # Padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self.get_indices(X.shape, HF, WF, stride, pad)
        # Multi-dimensional arrays indexing.
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def get_indices(self, X_shape, HF, WF, stride, pad):
        # get input size
        m, n_C, n_H, n_W = X_shape

        # get output size
        out_h = int((n_H + 2 * pad - HF) / stride) + 1
        out_w = int((n_W + 2 * pad - WF) / stride) + 1

        # ----Compute matrix of index i----

        # Level 1 vector.
        level1 = np.repeat(np.arange(HF), WF)
        # Duplicate for the other channels.
        level1 = np.tile(level1, n_C)
        # Create a vector with an increase by 1 at each level.
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        # Create matrix of index i at every levels for each channel.
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        # ----Compute matrix of index j----
        slide1 = np.tile(np.arange(WF), HF)
        slide1 = np.tile(slide1, n_C)
        everySlides = stride * np.tile(np.arange(out_w), out_h)
        # Create matrix of index j at every slides for each channel.
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

        # ----Compute matrix of index d----

        # This is to mark delimitation for each channel
        # during multi-dimensional arrays indexing.
        d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

        return i, j, d

    def col2im(self, dX_col, X_shape, HF, WF, stride, pad):
        # Transform our matrix back to the input image.
        # Get input size
        N, D, H, W = X_shape
        # Add padding if needed.
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, D, H_padded, W_padded))

        # Index matrices, necessary to transform our input image into a matrix.
        i, j, d = self.get_indices(X_shape, HF, WF, stride, pad)
        # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
        dX_col_reshaped = np.array(np.hsplit(dX_col, N))
        # Reshape our matrix back to image.
        # slice(None) is used to produce the [::] effect which means "for every elements".
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
        # Remove padding from new image if needed.
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[:, :, pad:-pad, pad:-pad]

    def step(self, batchsize, eta):
        self.kernels -= eta * self.grads / batchsize
        self.grads = 0

    def backward(self, dout):
        arr = []
        arr.append(dout)
        dout = np.array(arr)
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        # Compute bias gradient.
        self.b = np.sum(dout, axis=(0, 2, 3))
        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = self.col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        # Reshape dw_col into dw.
        # self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))
        dw = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))
        self.grads += dw

        return dX[0]


class Dropout():
    def __init__(self, p):
        self.p = p

    def forward(self, h):
        np.random.seed(2147483648)
        u = np.random.binomial(1, self.p, size=h.shape)
        h *= u
        return h

    def backward(self, h):
        return h

    def step(self, batchsize, eta):
        pass


class MaxPool():

    def __init__(self, filter_size, stride=2, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        # Apply Max pooling.
        self.cache = X
        arr = []
        arr.append(X)
        X = np.array(arr)

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = self.im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        M_pool = np.max(X_col, axis=1)
        # Reshape M_pool properly.
        M_pool = np.array(np.hsplit(M_pool, m))
        M_pool = M_pool.reshape(m, n_C, n_H, n_W)

        return M_pool[0]

    def step(self, batchsize, eta):
        pass

    def im2col(self, X, HF, WF, stride, pad):
        """
            Transforms our input image into a matrix.
            Parameters:
            - X: input image.
            - HF: filter height.
            - WF: filter width.
            - stride: stride value.
            - pad: padding value.
            Returns:
            -cols: output matrix.
        """
        # Padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self.get_indices(X.shape, HF, WF, stride, pad)
        # Multi-dimensional arrays indexing.
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def get_indices(self, X_shape, HF, WF, stride, pad):
        """
            Returns index matrices in order to transform our input image into a matrix.
            Parameters:
            -X_shape: Input image shape.
            -HF: filter height.
            -WF: filter width.
            -stride: stride value.
            -pad: padding value.
            Returns:
            -i: matrix of index i.
            -j: matrix of index j.
            -d: matrix of index d.
                (Use to mark delimitation for each channel
                during multi-dimensional arrays indexing).
        """
        # get input size
        m, n_C, n_H, n_W = X_shape

        # get output size
        out_h = int((n_H + 2 * pad - HF) / stride) + 1
        out_w = int((n_W + 2 * pad - WF) / stride) + 1

        # ----Compute matrix of index i----

        # Level 1 vector.
        level1 = np.repeat(np.arange(HF), WF)
        # Duplicate for the other channels.
        level1 = np.tile(level1, n_C)
        # Create a vector with an increase by 1 at each level.
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        # Create matrix of index i at every levels for each channel.
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        # ----Compute matrix of index j----

        # Slide 1 vector.
        slide1 = np.tile(np.arange(WF), HF)
        # Duplicate for the other channels.
        slide1 = np.tile(slide1, n_C)
        # Create a vector with an increase by 1 at each slide.
        everySlides = stride * np.tile(np.arange(out_w), out_h)
        # Create matrix of index j at every slides for each channel.
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

        # ----Compute matrix of index d----

        # This is to mark delimitation for each channel
        # during multi-dimensional arrays indexing.
        d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

        return i, j, d

    def backward(self, dout):
        """
            Distributes error through pooling layer.
            Parameters:
            - dout: Previous layer with the error.

            Returns:
            - dX: Conv layer updated with error.
        """
        X = self.cache
        arr = []
        arr.append(X)
        X = np.array(arr)
        arr2 = []
        arr2.append(dout)
        dout = np.array(arr2)

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        dout_flatten = dout.reshape(n_C, -1) / (self.f * self.f)
        dX_col = np.repeat(dout_flatten, self.f * self.f, axis=0)
        dX = self.col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        # Reshape dX properly.
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX[0]

    def col2im(self, dX_col, X_shape, HF, WF, stride, pad):
        """
            Transform our matrix back to the input image.
            Parameters:
            - dX_col: matrix with error.
            - X_shape: input image shape.
            - HF: filter height.
            - WF: filter width.
            - stride: stride value.
            - pad: padding value.
            Returns:
            -x_padded: input image with error.
        """
        # Get input size
        N, D, H, W = X_shape
        # Add padding if needed.
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, D, H_padded, W_padded))

        # Index matrices, necessary to transform our input image into a matrix.
        i, j, d = self.get_indices(X_shape, HF, WF, stride, pad)
        # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
        dX_col_reshaped = np.array(np.hsplit(dX_col, N))
        # Reshape our matrix back to image.
        # slice(None) is used to produce the [::] effect which means "for every elements".
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
        # Remove padding from new image if needed.
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]


class TanH():

    def __init__(self, alpha=1.7159):
        self.alpha = alpha
        self.cache = None

    def step(self, batchsize, eta):
        pass

    def forward(self, X):
        self.cache = X
        arr = [X]
        X = np.array(arr)
        return self.alpha * np.tanh(X)[0]

    def backward(self, new_deltaL):
        X = self.cache
        arr = [X]
        X = np.array(arr)
        return (new_deltaL * (1 - np.tanh(X) ** 2))[0]


class Relu():
    def __init__(self):
        self.relu_prime = lambda x: (x > 0) * 1
        self.x = None

    def forward(self, x):
        self.x = x
        scaling_factor = 1
        return np.maximum(0, x)  # * scaling_factor)

    def step(self, batchsize, eta):
        pass

    def backward(self, cache):
        scaling_factor = 0.1
        dX = cache.copy()
        dX[self.x <= 0] = 0
        dX[self.x > 0] *= scaling_factor
        return dX

    # class Relu():
    #     def __init__(self):
    #         self.relu_prime = lambda x: (x > 0) * 1
    #         self.x = None
    #
    #     def forward(self, image):
    #         self.x = image
    #         relu_out = np.zeros(image.shape)
    #         for map_num in range(image.shape[-1]):
    #             for r in np.arange(0, image.shape[0]):
    #                 for c in np.arange(0, image.shape[1]):
    #                     relu_out[r, c, map_num] = np.max([image[r, c, map_num], 0])
    #         return relu_out

    def step(self, batchsize, eta):
        pass

    def backward(self, cache):
        cache = cache * self.relu_prime(self.x)
        return cache


class Softmax():
    def forward(self, x):
        y = np.exp(x)
        self.h = y / y.sum(axis=None, keepdims=True) + 1e-15
        return self.h

    def backward(self, y):
        return self.h - y

    def step(self, batchsize, eta):
        pass


class LinearRelu():
    def __init__(self):
        self.x = None

    def relu_prime(self, x):
        return np.where(x > 0, 1.0, 0.0)

    def forward(self, input):
        self.input = input
        input[input < 0] = 0
        return input.copy()

    def backward(self, cache):
        cache = cache * self.relu_prime(self.input)
        return cache

    def step(self, batchsize, eta):
        pass


class leaky():
    def forward(self, x):
        leaky_way1 = np.where(x > 0, x, x * 0.01)
        return leaky_way1

    def backward(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def step(self, batchsize, eta):
        pass


class Linear():
    def __init__(self, in_size, out_size, uniform_range):
        self.W = np.random.uniform(-uniform_range, uniform_range, ([in_size, out_size]))
        self.b = np.random.uniform(-uniform_range, uniform_range, [1, out_size])
        self.x = 0
        self.z = 0
        self.grad = 0
        self.b_grad = 0

    def forward(self, input):
        self.input = input
        self.z = (np.dot(input, self.W) + self.b)
        return self.z

    def backward(self, cache):
        self.grad += np.dot(cache.T, self.input).T
        self.b_grad += cache
        cache = np.dot(cache, self.W.T)
        return cache

    def step(self, batchsize, eta):
        self.b -= eta * self.b_grad / batchsize
        self.W -= eta * self.grad / batchsize
        self.grad = 0
        self.b_grad = 0


class Flatten():
    def forward(self, image):
        self.image_shape = image.shape
        flat = image.flatten()
        flat = np.array([flat])
        return flat

    def step(self, batchsize, eta):
        pass

    def backward(self, dout):
        out = dout.reshape(self.image_shape)
        return out
