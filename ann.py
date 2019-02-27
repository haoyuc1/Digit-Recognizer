import numpy as np

class ANN:
    def __init__(self, num_input, num_hidden, num_output):
        # Number of nodes in input, hidden, and output layers
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.weight_input = np.random.randn(self.num_input, self.num_hidden) * np.random.choice([-1, 1], (self.num_input, self.num_hidden))
        # print(self.weight_input)
        self.weight_hidden = np.random.randn(self.num_hidden, self.num_output) * np.random.choice([-1, 1], (self.num_hidden, self.num_output))
        # print(self.weight_hidden)

        self.bias_hidden = np.random.randn(self.num_input, 1) * np.random.choice([-1, 1], (self.num_input, 1))
        # print(self.bias_hidden)
        self.bias_output = np.random.randn(self.num_hidden, 1) * np.random.choice([-1, 1], (self.num_hidden, 1))
        # print(self.bias_output)

        self.activation_hidden = []
        self.activation_output = []


    # sigmoid threshold function 1/(1+e^-x)
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # derivative of the sigmoid function
    def dsigmoid(self, y):
        return y * (1.0 - y)

    def activation(self, a, w, b):
        return self.sigmoid(sum(np.dot(a, w)) + b[0])

    def forward_propagate(self, input):
        if len(input) != self.num_input:
            raise ValueError('Incorrect number of inputs.')
        for i in range(self.num_hidden):
            self.activation_hidden.append(self.activation(float(input[i]), self.weight_input[i], self.bias_hidden[i]))
        print(self.activation_hidden)
        for j in range(self.num_output):
            self.activation_output.append(self.activation(self.activation_hidden[j], self.weight_hidden[j], self.bias_output[j]))
        print(self.activation_output)

    def back_propagation(self, output):
        return
