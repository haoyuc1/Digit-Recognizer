import csv
from funcNN import *
for _ in range(10):
    train = open("./csv/train.csv","r")
    train.readline()
    cost = 0
    m = 0
    for line in csv.reader(train):
        ip = line
    #ip = train.readline()
        y  = int(ip[0])
        ip = [float(i)/255 for i in ip[1:]]
        ip = np.matrix(ip)
        ip = np.array(ip)
        output = forward_propagate(ip)
        cost += np.sum(error(vectorprediction(y),output))
        m += 1
        back_propagate(y,output,ip)
    print(cost/m)
    update(m)
train = open("./csv/train.csv","r")
train.readline()
for line in csv.reader(train):
    ip = line
    y  = int(ip[0])
    ip = [float(i)/255 for i in ip[1:]]
    ip = np.matrix(ip)
    ip = np.array(ip)
    output = forward_propagate(ip)
    print(y,output)


