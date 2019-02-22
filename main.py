#%%
import csv
import numpy
import ann
# from matplotlib import pyplot as plt

def main():
    input_layer_weight = numpy.random.rand(785,15)*numpy.random.choice([-1,1], (785,15))
    hiden_layer_weight = numpy.random.rand(16,10)*numpy.random.choice([-1,1], (16,10))

    def sigmoid(Z):
        return 1/(1+numpy.exp(-Z))

    def result(output):
        return output.index(max(output))



    #%% Train
    with open("csv/train.csv") as file:
        file.readline()
        train = csv.reader(file)
        for line in train:
        #     # print(line)
        #     label = int(line[0])
        #     expect = ([0]*10)
        #     expect[label] = 1
        #     pixel = [[int(i) for i in line[1:]]]
        #     # plt.imshow(numpy.reshape(numpy.asarray(pixel),(28,28)))
        #     # plt.show()
        #     Input = numpy.insert(numpy.asarray(pixel)/255,0,1,axis = 1)
        #     # print(Input)
        #     Output = sigmoid(numpy.dot(numpy.insert(numpy.dot(Input,input_layer_weight),0,1,axis = 1),hiden_layer_weight))
        #     # print(Output)
        #     # print(result(Output.tolist()[0]))

            network = ann.ANN(784, 15, 10)
            network.feed_forward(line[1:])
            break

#%%

if __name__ == "__main__":
    main()
