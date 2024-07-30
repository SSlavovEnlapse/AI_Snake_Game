import numpy as np

def relu(x):
    return np.maximum(0, x)

class NeuralNetwork:
    def __init__(self, input_size=24, hidden_layer_sizes=[16, 16], output_size=4):
        self.layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]

    def forward(self, x):
        activation = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = np.maximum(0, z)  # ReLU activation
        return activation

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output)