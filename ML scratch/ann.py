import numpy as np


class DenseLayer:
    def __init__(self, input_dim, output_dim, l2_lambda=0.001,
                 he_init=True):
        if he_init:
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(
                2.0 / input_dim
            )
        else:
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(
                1.0 / input_dim
            )

        self.b = np.zeros((1, output_dim))
        self.l2_lambda = l2_lambda

        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        # dZ already contains the batch averaging from the loss.
        self.dW = self.X.T @ dZ + self.l2_lambda * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # Gradient passed to the previous layer.
        dX = dZ @ self.W.T
        return dX


class ReLU:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)


class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        # Mean over examples and output dimensions.
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        n = self.y_true.size
        return 2.0 * (self.y_pred - self.y_true) / n


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient):
        grad = loss_gradient

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db


def regularization_loss(model):
    total = 0.0

    for layer in model.layers:
        if isinstance(layer, DenseLayer):
            total += 0.5 * layer.l2_lambda * np.sum(layer.W ** 2)

    return total


def train(model, X, y, epochs=1000, lr=0.01, batch_size=32):
    loss_fn = MSELoss()
    n_samples = X.shape[0]

    for epoch in range(epochs):
        # Shuffle the training data every epoch.
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0.0

        for start in range(0, n_samples, batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            y_pred = model.forward(X_batch)

            # Calculate data loss
            data_loss = loss_fn.forward(y_pred, y_batch)

            # Include L2 penalty in the reported objective
            loss = data_loss + regularization_loss(model)

            # Backpropagation
            loss_gradient = loss_fn.backward()
            model.backward(loss_gradient)

            # Gradient descent
            model.update(lr)

            epoch_loss += data_loss * len(X_batch)

        epoch_loss /= n_samples

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Data Loss: {epoch_loss:.6f}, "
                f"Total Loss: {epoch_loss + regularization_loss(model):.6f}"
            )


# Build the neural network

np.random.seed(42)

model = NeuralNetwork()

model.add(DenseLayer(10, 64, l2_lambda=0.001))
model.add(ReLU())

model.add(DenseLayer(64, 32, l2_lambda=0.001))
model.add(ReLU())

model.add(DenseLayer(32, 1, l2_lambda=0.001, he_init=False))


# Example training data


X = np.random.randn(1000, 10)
y = (
    3 * X[:, 0:1]
    - 2 * X[:, 1:2]
    + 0.5 * X[:, 2:3]
    + 0.1 * np.random.randn(1000, 1)
)

train(model, X, y, epochs=1000, lr=0.01, batch_size=32)

# Predictions
predictions = model.forward(X[:5])
print("Predictions:\n", predictions)
print("Actual values:\n", y[:5])