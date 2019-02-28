import numpy as np

class ANN:
    def __init__(self, num_input, num_hidden, num_output, learning_rate = 0.2, threshold = 0.97):
        # Number of nodes in input, hidden, and output layers
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.alpha = learning_rate
        self.threshold = threshold

        self.w_input = np.random.randn(self.num_input, self.num_hidden) * np.random.choice([-1, 1], (self.num_input, self.num_hidden))
        # print(self.w_input)
        self.w_hidden = np.random.randn(self.num_hidden, self.num_output) * np.random.choice([-1, 1], (self.num_hidden, self.num_output))
        # print(self.w_hidden)

        self.b_hidden = np.random.randn(1, self.num_hidden) * np.random.choice([-1, 1], (1, self.num_hidden))
        # print(self.b_hidden)
        self.b_output = np.random.randn(1, self.num_output) * np.random.choice([-1, 1], (1, self.num_output))
        # print(self.b_output)
        
        self.a_input = []
        self.a_hidden = []
        self.a_output = []

        self.d_input = []
        self.d_hidden = []
        self.d_output = []

    # sigmoid threshold function 1/(1+e^-x)
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # derivative of the sigmoid function
    def d_sigmoid(self, y):
        return y * (1.0 - y)

    def activation(self, a, w, b):
        return self.sigmoid(np.dot(a, w) + b)

    def d_activation(self, a, w, b):
        return self.d_sigmoid(np.dot(a, w) + b)

    def forward_propagate(self, input):
        self.a_input = [np.array(input)]
        self.a_hidden = self.activation(self.a_input, self.w_input, self.b_hidden)
        self.a_output = self.activation(self.a_hidden, self.w_hidden, self.b_output)

    def back_propagate(self, input, output):
        actual = [0] * 10
        actual[output] = 1
        while True:
            self.forward_propagate(input)
            print(self.a_output)
            # 终止条件 - 不够全面需要修改
            if self.a_output[0][output] > self.threshold:
                break
            # 计算delta
            self.d_output = np.array([x - y for x, y in zip(actual, self.a_output)])
            self.d_hidden =  self.d_activation(self.a_hidden, self.w_hidden, self.b_output) * self.d_output
            self.d_input = self.d_activation(self.a_input, self.w_input, self.b_hidden) * sum([x * y for x , y in zip(self.w_input, self.d_hidden[0])])
            # 更新weights
            self.w_input = self.w_input + self.alpha * np.dot(self.d_input.T, self.a_input).T
            self.w_hidden = self.w_hidden + self.alpha * np.dot(self.d_hidden.T, self.a_hidden).T
