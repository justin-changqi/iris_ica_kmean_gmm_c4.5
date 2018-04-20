import pandas as pd
import numpy as np
import math

class IrisGMM:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.petal_length = np.array(list(map(list, zip(*self.irisdata)))[2])
        # print (self.petal_length)

    def applyGmm(self, init_means, init_variance, alpha, iterative=3000, num_components=3):
        self.means = np.array(init_means)
        self.variance = np.array(init_variance)
        self.alpha = np.array(alpha)
        self.weights = np.zeros((len(self.petal_length), num_components))
        for i in range(iterative):
            self.E_step()
            self.M_step()

    def E_step(self):
        weight_shape = self.weights.shape
        for i in range(weight_shape[0]):
            weight_k_sum = 0
            for j in range(weight_shape[1]):
                # Gaussian with one dimenssion format
                self.weights[i][j] = self.alpha[j] * math.exp(-pow(self.petal_length[i] - self.means[j], 2) / (2*self.variance[j])) \
                                / math.sqrt(2*math.pi*self.variance[j])
                weight_k_sum += self.weights[i][j]
            self.weights[i] /= weight_k_sum
        # print (self.weights)

    def M_step(self):
        shape = self.weights.shape
        # Update alpha and Means
        alpha_sum = 0.
        Nk = np.zeros(shape[1])
        for k in range(shape[1]):
            self.alpha[k] = 0.
            self.means[k] = 0.
            for i in range(shape[0]):
                self.alpha[k] += self.weights[i][k]
                self.means[k] += (self.weights[i][k] * self.petal_length[i])
            Nk[k] = self.alpha[k]
            alpha_sum += Nk[k]
            self.means[k] /= Nk[k]
        self.alpha /= alpha_sum
        # Update variance
        for k in range(shape[1]):
            self.variance[k] = 0.
            for i in range(shape[0]):
                self.variance[k] += (self.weights[i][k] * pow(self.petal_length[i] - self.means[k], 2))
            self.variance[k] /= Nk[k]
        # print ('Update Mean: ', self.means)
        # print ('Update Alpha: ', self.alpha)
        # print ('Update Variance: ', self.variance)

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    iris_gmm = IrisGMM('iris_data_set/iris.data')
    iris_gmm.applyGmm([1., 4., 6.], [1., 1., 1.], [0.5, 0.25, 0.25],  iterative=3000, num_components=3)
    print ('Means: ', iris_gmm.means)
    print ('iris Means: ', [iris_gmm.petal_length[:50].sum()/150, iris_gmm.petal_length[50:100].sum()/150, iris_gmm.petal_length[100:150].sum()/150])
