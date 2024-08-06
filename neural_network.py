# neural_network.py

import numpy as np
import json

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

    def save(self, filename):
        model_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]