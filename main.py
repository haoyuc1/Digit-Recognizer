import csv
import numpy
import ann
import matplotlib.pyplot as plt


def showresult(result):
    result = result.tolist()
    print(result)
    print(result.index(max(result)))


def process_csv(func, path, *args):
    with open(path) as file:
        file.readline()
        data = csv.reader(file)
        ind = 0
        for row in data:
            print(ind)
            ind+=1
            func(row, *args)


def train(example, network):
    input = [float(i) / 255.0 for i in example[1:]]
    output = int(example[0])
    predict = network.forward_propagate(input)
    label = [0.0]*network.layers[-1].size
    label[output]=1.0
    network.back_propagate(label)


def recognize(item, network):
    #input = [float(i) / 255.0 for i in item]
    input = [float(i) / 255.0 for i in item[1:]]
    predict = network.forward_propagate(input)
    pixel = [input]
    plt.imshow(numpy.reshape(numpy.asarray(pixel),(28,28)))
    plt.show()
    showresult(predict)
    print(item[0])


def main():
    network = ann.DNN([784, 15, 10])
    #process_csv(recognize, "csv/test.csv", network)
    for _ in range(10):
        error = []
        process_csv(train, "csv/train.csv", network)
        print(network.cost())
        network.gradientDescent()
        network.resetD()
    #process_csv(recognize, "csv/train.csv", network)


if __name__ == "__main__":
    main()
