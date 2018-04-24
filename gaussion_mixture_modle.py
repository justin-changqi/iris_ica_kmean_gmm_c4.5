import pandas as pd
import numpy as np
import math

class IrisGMM:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' \
                                                    else (1 if x == 'Iris-versicolor' else 2))
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
                a = self.alpha[j]
                x = self.petal_length[i]
                m = self.means[j]
                var = self.variance[j]
                PI = math.pi
                self.weights[i][j] = a * (math.exp(-pow(x - m, 2) / (2*var)) / math.sqrt(2*PI*var))
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

    def classify(self, test_data):
        self.clusters = {}
        num_cluster = len(self.means)
        PI = math.pi
        for i in range(len(test_data)):
            max_p = {'cluter':None, 'probability':-1}
            x = test_data[i]
            for j in range(num_cluster):
                m = self.means[j]
                var = self.variance[j]
                p = math.exp(-pow(x - m, 2) / (2*var)) / math.sqrt(2*PI*var)
                if (p > max_p['probability']):
                     max_p['probability'] = p
                     max_p['cluter'] = j
            if max_p['cluter'] not in self.clusters:
                self.clusters.update({max_p['cluter']:[i]})
            else:
                self.clusters[max_p['cluter']].append(i)
        # print (self.clusters)

    def getAccuracy(self, cluster):
        cluster_list = {}
        indexis = self.clusters[cluster]
        for index in indexis:
            key = self.irisdata[index][-1]
            if key in cluster_list:
                cluster_list[key] += 1.
            else:
                cluster_list.update({int(key):1.})
        # get main key
        max_item = {'key':None, 'amount':-1}
        for key in cluster_list:
            if cluster_list[key] > max_item['amount']:
                 max_item['amount'] = cluster_list[key]
                 max_item['key'] = key
        num_wrong = 0
        for key in cluster_list:
            if key !=  max_item['key']:
                num_wrong += cluster_list[key]
        # print (cluster_list,  max_item['key'], int(num_wrong) )
        return  cluster_list, int(num_wrong)

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    iris_gmm = IrisGMM('iris_data_set/iris.data')
    iris_gmm.applyGmm([1., 4., 6.], [1., 1., 1.], [0.5, 0.25, 0.25],  iterative=3000, num_components=3)
    print ('Means: ', iris_gmm.means)
    print ('Variance: ', iris_gmm.variance)
    print ('Alpha: ', iris_gmm.alpha)
    print ('============ Cluster Test Results ============')
    iris_gmm.classify(iris_gmm.petal_length)
    for key in iris_gmm.clusters:
        print ('cluster', key, ':')
        print ('\t(c) Number of member in cluster -->',len(iris_gmm.clusters[key]))
        cluster_list, num_wrong = iris_gmm.getAccuracy(key)
        print ('\t    Statistic -->', cluster_list)
        print ('\t    Number of Wrong -->', num_wrong)
