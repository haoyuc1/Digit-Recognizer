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
        for row in data:
            func(row, *args)


def train(example, network, error):
    input = [float(i) / 255.0 for i in example[1:]]
    output = int(example[0])
    predict = network.forward_propagate(input)
    error.append(network.error(predict, output))
    network.back_propagate(output)


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
    network = ann.ANN(784, 15, 10)
    #process_csv(recognize, "csv/test.csv", network)
    for i in range(10):
        error = []
        process_csv(train, "csv/train.csv", network, error)
        D_hidden = network.d_hidden/len(error)
        #print(D_hidden)
        D_input = network.d_input/len(error)
        #print(D_input)
        cost = sum(error)/len(error)
        print(cost)
        network.gradientDescent(D_input, D_hidden)
        network.resetD()
    process_csv(recognize, "csv/train.csv", network)


if __name__ == "__main__":
    main()
