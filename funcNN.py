import numpy as np
epsilon_init = 0.12
alpha = 0.2
def randWeight(weight):
    return np.random.rand(weight.shape[0],weight.shape[1])*2*epsilon_init-epsilon_init
inputLayer = np.zeros((1,784))
hiddenLayer = np.zeros((1,15))
inputWeight = np.ones((785,15))
hiddenWeight = np.ones((16,10))
inputWeight = randWeight(inputWeight)
hiddenWeight = randWeight(hiddenWeight)
d_hiddenLayer = np.zeros((1,15))
d_outputLayer = np.zeros((1,10))
d_inputWeight = np.zeros((785,15))
d_hiddenWeight = np.zeros((16,10))
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def vectorprediction(y):
    vector = np.zeros((1,10))
    vector[0][y] = 1
    return vector

def forward(layer,weight):
    return sigmoid(np.dot(np.append(layer,[[1]],axis=1),weight))

def error(prediction, layer):
    return -prediction*np.log(layer) - (1-prediction)*np.log(1-layer)

def backward(layer,weight,player):
    return np.dot(layer,np.transpose(np.delete(weight,-1,axis=0)))*player*(1-player)

def gradient(layer,d_layer):
    return np.dot(np.transpose(np.append(layer,[[1]],axis=1)),d_layer)

def update(m):
    global inputWeight
    global hiddenWeight
    global d_inputWeight
    global d_hiddenWeight

    inputWeight -= alpha*d_inputWeight/m
    hiddenWeight -= alpha*d_hiddenWeight/m
    d_inputWeight = np.zeros((785,15))
    d_hiddenWeight = np.zeros((16,10))


def forward_propagate(ip):
    global inputLayer
    global hiddenLayer
    global inputWeight
    global hiddenWeight
    if ip.shape == inputLayer.shape:
        inputLayer = ip
    else:
        raise Exception
    hiddenLayer = forward(inputLayer,inputWeight)
    output = forward(hiddenLayer,hiddenWeight)
    return output

def back_propagate(y,output,ip):
    global d_outputLayer
    global d_hiddenLayer
    global hiddenWeight
    global hiddenLayer
    global d_inputWeight
    global d_hiddenWeight
    d_outputLayer = output - vectorprediction(y)
    d_hiddenLayer = backward(d_outputLayer,hiddenWeight,hiddenLayer)
    d_hiddenWeight += gradient(hiddenLayer,d_outputLayer)
    d_inputWeight += gradient(ip,d_hiddenLayer)
    return

    

# def Mplus(vector1,vector2):
#     pass

# def Splus(vector1,num):
#     pass

# def Msubstraction(vector1,vector2):
#     pass

# def Ssubstraction(vector1,num):
#     pass

# def Mproduct(vector1,vector2):
#     pass

# def Sproduct(vector1,num):
#     pass

# def dotproduct(vector1,vector2):
#     pass

