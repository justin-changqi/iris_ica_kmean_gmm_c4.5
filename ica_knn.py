import pandas as pd
import numpy as np
import random
import math
from numpy import linalg as LA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

from knn import Knn

class IrisICA:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' \
                                                    else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.train_data = []
        self.test_data = []
        self.number_of_features = len(self.irisdata[0]) - 1

    def plotIrisData(self, title):
        plt.figure(title)
        plot_data = {}
        for data in self.irisdata:
            if data[-1] not in plot_data:
                plot_data.update({data[-1]: [data[:-1]]})
            else:
                plot_data[data[-1]].append(data[:-1])
        # print (plot_data)
        plot_data_T = []
        for key in plot_data:
            plot_data_T.append(list(map(list, zip(*plot_data[key]))))
        n_feature = self.number_of_features
        for i in range(n_feature):
            for j in range(n_feature):
                plt.subplot(n_feature, n_feature, (i*n_feature)+j+1)
                if (i != j):
                    plt.plot(plot_data_T[0][i], plot_data_T[0][j], 'ro', \
                    plot_data_T[1][i], plot_data_T[1][j], 'bo', \
                    plot_data_T[2][i], plot_data_T[2][j], 'go')
                else:
                    plt.plot()

    def applyIcaFromFullIris(self, number_components=4):
        ica_input = []
        for data in self.irisdata:
            ica_input.append(data[:-1])
        # print (np.array(ica_input))
        ica = FastICA(n_components=number_components, whiten=False)
        ica_out = ica.fit_transform(ica_input)
        # replace original data
        for data_iris, ica_data in zip(self.irisdata, ica_out):
            data_iris[:-1] = ica_data
        # print (np.array(self.irisdata))

    def getSortedComponentEnergy(self):
        energies = []
        num_conponent = len(self.irisdata[0])-1
        for data in self.irisdata:
            for i in range(num_conponent):
                if len(energies) < num_conponent:
                    energies.append(math.pow(data[i], 2))
                else:
                    energies[i] += math.pow(data[i], 2)
        sorted_engergy_index = sorted(range(len(energies)), \
                                      key=lambda x:energies[x], reverse=True)
        # print (sorted_engergy_index)
        # print (energies)
        return sorted_engergy_index

    def getTrainTestSet(self, components_index, train_size=0.7):
        random.shuffle(self.irisdata)
        num_train = int(len(self.irisdata) * train_size)
        for i in range(len(self.irisdata)):
            data_point = []
            for index in components_index:
                data_point.append(self.irisdata[i][index])
            data_point.append(self.irisdata[i][-1])
            if (i <= num_train):
                self.train_data.append(data_point)
            else:
                self.test_data.append(data_point)
        return self.train_data, self.test_data

def icaKnnTest():
    iris_data = IrisICA('iris_data_set/iris.data')
    iris_data.plotIrisData('iris data before ica')
    iris_data.applyIcaFromFullIris(number_components=4)
    energy_of_components = iris_data.getSortedComponentEnergy()
    train_data, test_data = iris_data.getTrainTestSet(energy_of_components[:2], train_size=0.7)
    iris_data.plotIrisData('iris data after ica')
    knn = Knn()
    print (knn.kNearestNeighbors(train_data, test_data))
    plt.show()

def icaKnnLoop(loop=10):
    accuracy = 0
    for i in range(loop):
        iris_data = IrisICA('iris_data_set/iris.data')
        iris_data.applyIcaFromFullIris(number_components=4)
        energy_of_components = iris_data.getSortedComponentEnergy()
        train_data, test_data = iris_data.getTrainTestSet(energy_of_components[:2], train_size=0.7)
        knn = Knn()
        current_accuracy = knn.kNearestNeighbors(train_data, test_data)
        accuracy += current_accuracy
        print ('round ', i+1, ' accuracy: ', current_accuracy)
    return accuracy / loop

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    print ('Average accuracy: ', icaKnnLoop(loop=10))
