import numpy as np

class ANN:
    def __init__(self, num_input, num_hidden, num_output, learning_rate = 0.2, threshold = 0.97, lamba = 10):
        # Number of nodes in input, hidden, and output layers
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.alpha = learning_rate
        self.lamba = lamba
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

        self.d_input = np.zeros((self.num_input, self.num_hidden))
        self.d_hidden = np.zeros((self.num_hidden, self.num_output))


    # sigmoid threshold function 1/(1+e^-x)
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def error(self, h, tag):
        y = np.zeros((self.num_output,))
        y[tag] = 1
        h = h[0]
        return -sum(y*np.log(h)+(1-y)*np.log(1-h))

    # derivative of the sigmoid function
    def d_sigmoid(self, a):
        return a * (1.0 - a)

    def activation(self, a, w, b):
        return self.sigmoid(np.dot(a, w) + b)

    def d_activation(self, a, w, b):
        return self.d_sigmoid(np.dot(a, w) + b)

    def forward_propagate(self, input):
        self.a_input = [np.array(input)]
        self.a_hidden = self.activation(self.a_input, self.w_input, self.b_hidden)
        self.a_output = self.activation(self.a_hidden, self.w_hidden, self.b_output)
        return self.a_output

    def back_propagate(self, label):
        actual = np.zeros((self.num_output,))
        actual[label] = 1
        delta3 = self.a_output - actual
        delta2 = np.dot(self.w_hidden,delta3.transpose()).transpose()*self.a_hidden*(1-self.a_hidden)
        self.d_hidden += np.dot(self.a_hidden.transpose(),delta3)
        self.d_input += np.dot(np.array(self.a_input).transpose(),delta2)
        # actual = [0] * 10
        # actual[output] = 1
        # while True:
        #     self.forward_propagate(input)
        #     print(self.a_output)
        #     # 终止条件 - 不够全面需要修改
        #     if self.a_output[0][output] > self.threshold:
        #         break
        #     # 计算delta
        #     self.d_output = np.array([x - y for x, y in zip(actual, self.a_output)])
        #     self.d_hidden =  self.d_activation(self.a_hidden, self.w_hidden, self.b_output) * self.d_output
        #     self.d_input = self.d_activation(self.a_input, self.w_input, self.b_hidden) * sum([x * y for x , y in zip(self.w_input, self.d_hidden[0])])
        #     # 更新weights
        #     self.w_input = self.w_input + self.alpha * np.dot(self.d_input.T, self.a_input).T
        #     self.w_hidden = self.w_hidden + self.alpha * np.dot(self.d_hidden.T, self.a_hidden).T
    def gradientDescent(self,D_input,D_hidden):
        self.w_input-=self.alpha*D_input
        self.w_hidden-=self.alpha*D_hidden
    def resetD(self):
        self.d_input = np.zeros((self.num_input, self.num_hidden))
        self.d_hidden = np.zeros((self.num_hidden, self.num_output))