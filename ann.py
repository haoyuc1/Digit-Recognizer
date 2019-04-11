import numpy as np
import random
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class Weight:
    def __init__(self, size):
        self.size = size
        self.vals = []
        self.cumulation = [0]*size
        self.symmetryBreaking()

    def symmetryBreaking(self):
        epsilon_init = 0.12
        self.vals = [random.random() * 2 * epsilon_init - epsilon_init for _ in range(self.size)]

    def resetD(self):
        self.cumulation = [0]*self.size

    def update(self,alpha):
        self.vals = [self.vals[i] - alpha*self.cumulation[i] for i in range(self.size)]

class Layer:
    def __init__(self, size):
        self.size = size
        self.vals = [0]*size
        self.dvals = [0]*size
        self.connected = {}
        self.connected_back = {}

    def input(self, vals):
        if len(vals) == self.size:
            self.vals = vals
        else:
            raise TypeError('Size {} not fix layer size {}'.format(len(vals),self.size))
        
    def feedback(self, dvals):
        if len(dvals) == self.size:
            self.dvals = dvals
        else:
            raise TypeError('Size {} not fix layer size {}'.format(len(dvals),self.size))
    
    def connect(self,layer):
        if layer not in self.connected:
            W_ls = [Weight(layer.size) for _ in range(self.size+1)]
            self.connected[layer] = W_ls
            layer.connect_back(self)
        return self.connected[layer]

    def connect_back(self,layer):
        if layer not in self.connected_back:
            self.connected_back[layer] = layer.connected[self]
        return self.connected_back[layer]

    
    def forward(self,layer):
        if layer not in self.connected:
            self.connect(layer)
        W_ls = self.connected[layer]
        update_ls = [sigmoid(sum([W_ls[i].vals[j]*n for i,n in enumerate(self.vals+[1])])) for j in range(layer.size)]         
        layer.input(update_ls)
        #print(W_ls[0].vals[0])
        return layer
    
    def backward(self,layer):
        if layer not in self.connected_back:
            raise TypeError("This layer haven\'t be connected")
        W_ls = self.connected_back[layer]
        update_ls = [sum([self.dvals[i]*W_ls[w_i].vals[i] for i in range(len(W_ls[w_i].vals))])*layer.vals[w_i]*(1-layer.vals[w_i]) for w_i in range(len(W_ls)-1)]              
        layer.feedback(update_ls)
        layer1 = layer.vals+[1]
        for i in range(len(W_ls)):
            for j in range(len(W_ls[i].vals)):
                W_ls[i].cumulation[j] += layer1[i]*self.dvals[j]

        return layer

    def output(self):
        return self.vals

    def update(self,alpha):
        for i in self.connected:
            for w in self.connected[i]:
                w.update(alpha)
    
    def resetD(self):
        for i in self.connected:
            for w in self.connected[i]:
                w.resetD()



class DNN:
    def __init__(self, layers_size, learning_rate = 0.2, threshold = 0.97, lamda = 10):
        self.alpha = learning_rate
        self.lamda = lamda
        self.threshold = threshold

        self.layers = []
        for i in layers_size:
            self.layers.append(Layer(i))

        self.m = 0
        self.error = 0
        
    def forward_propagate(self,ip_data):
        pre = None
        for i,L in enumerate(self.layers):
            if pre:
                pre.forward(L)
            else:
                L.input(ip_data)
            pre = L
        return L.output()

    def back_propagate(self,label):
        if len(label) != self.layers[-1].size:
            raise TypeError('Label size {} not fix output layer size {}'.format(len(label),self.layers[-1].size))
        else:
            self.layers[-1].feedback([self.layers[-1].vals[j] - label[j] for j in range(len(label))])
            for i in reversed(range(1,len(self.layers))):
                self.layers[i].backward(self.layers[i-1])
        self.m+=1
        #try:
        er = []
        for i in range(len(label)):
            first = -label[i]*math.log(self.layers[-1].vals[i])
            #try:
            second = -(1-label[i])*math.log(1-self.layers[-1].vals[i])
            #except:
            #    print(self.layers[-1].vals[i])
            er.append(first+second)
        self.error+=sum(er)
        # self.error += sum([-(label[i]*math.log(self.layers[-1].vals[i])+(1-label[i])*math.log(1-self.layers[-1].vals[i])) for i in range(len(label))])
        #except Exception as e:

    def gradientDescent(self):
        for i in self.layers:
            i.update(self.alpha)

    def resetD(self):
        for i in self.layers:
            i.resetD()
    
    def cost(self):
        return self.error/self.m
        

class ANN:
    def __init__(self, num_input, num_hidden, num_output, learning_rate = 0.2, threshold = 0.97, lamba = 10):
        # Number of nodes in input, hidden, and output layers
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.alpha = learning_rate
        self.lamba = lamba
        self.threshold = threshold

        self.w_input = np.random.randn(self.num_input+1, self.num_hidden) * np.random.choice([-1, 1], (self.num_input+1, self.num_hidden))
        # print(self.w_input)
        self.w_hidden = np.random.randn(self.num_hidden+1, self.num_output) * np.random.choice([-1, 1], (self.num_hidden+1, self.num_output))
        # print(self.w_hidden)

        self.b_hidden = np.random.randn(1, self.num_hidden) * np.random.choice([-1, 1], (1, self.num_hidden))
        # print(self.b_hidden)
        self.b_output = np.random.randn(1, self.num_output) * np.random.choice([-1, 1], (1, self.num_output))
        # print(self.b_output)
        
        self.a_input = []
        self.a_hidden = []
        self.a_output = []

        self.d_input = np.zeros((self.num_input+1, self.num_hidden))
        self.d_hidden = np.zeros((self.num_hidden+1, self.num_output))


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

    def activation(self, a, w):
        return self.sigmoid(np.dot(a, w))

    def d_activation(self, a, w):
        return self.d_sigmoid(np.dot(a, w))

    def forward_propagate(self, input):
        self.a_input = [np.array(input+[1])]
        self.a_hidden = self.activation(self.a_input, self.w_input)
        self.a_hidden = np.array(np.append(self.a_hidden.transpose(),[[1]]).transpose())
        self.a_output = self.activation(self.a_hidden, self.w_hidden)
        return self.a_output

    def back_propagate(self, label):
        actual = np.zeros((self.num_output,))
        actual[label] = 1
        delta3 = self.a_output - actual
        delta2 = np.dot(self.w_hidden,delta3.transpose()).transpose()*self.a_hidden*(1-self.a_hidden)
        self.d_hidden += np.dot(self.a_hidden[:,None],delta3[None,:])
        self.d_input += np.dot(np.array(self.a_input).transpose(),delta2[None,0:15])
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
        self.w_input-=self.alpha*D_input[:,0:15]
        self.w_hidden-=self.alpha*D_hidden
    def resetD(self):
        self.d_input = np.zeros((self.num_input+1, self.num_hidden))
        self.d_hidden = np.zeros((self.num_hidden+1, self.num_output))
