import csv
import numpy
import ann
# from matplotlib import pyplot as plt


def process_csv(func,path,*args):
    with open(path) as file:
        file.readline()
        data = csv.reader(file)
        for row in data:
            func(row,*args)


def train(example,network):
    network.forward_propagate(example[1:])

def recognize(item,network):
    network.forward_propagate(item)

# def train():
#     with open("csv/train.csv") as training_set:
#         training_set.readline()
#         training_examples = csv.reader(training_set)
#         for example in training_examples:
#             network = ann.ANN(784, 15, 10)
#             network.feed_forward(example[1:])

# def recognize():
#     with open("csv/test.csv") as test_set:
#         test_set.readline()
#         test_items = csv.reader(test_set)
#         for item in test_items:
#             network.feed_forward(item[1:])


def main():
    network = ann.ANN(784, 15, 10)
    process_csv(train,"csv/train.csv",network)




if __name__ == "__main__":
    main()
