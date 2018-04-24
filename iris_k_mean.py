import pandas as pd
import numpy as np
import math

class IrisKMeans:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' \
                                                    else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.clusters = {}
        self.number_of_features = len(self.irisdata[0]) - 1

    def getFirstSampleIndex(self):
        index = []
        classes = []
        for i in range(len(self.irisdata)):
            class_exist = False
            for _class in classes:
                if _class == self.irisdata[i][-1]:
                    class_exist = True
            if not class_exist:
                index.append(i)
                classes.append(self.irisdata[i][-1])
        # print (index)
        return index

    def kMeansClustering(self, init_points_index, k=3):
        self.clusters = {}
        self.last_means = {}
        iterative = 0
        # update initial points
        for i in range(len(init_points_index)):
            index = init_points_index[i]
            self.clusters.update({i:{'mean':self.irisdata[index][:-1], 'cluster_indexes':[index]}})
        # print (self.clusters)
        # repeat clustering util convention
        while True:
            # clear indexes in each classes
            for key in self.clusters:
                self.clusters[key]['cluster_indexes'] = []
            # clustering with new mean
            for i in range(len(self.irisdata)):
                # find the point is near which mean
                shortest_cluster = {}
                for key in self.clusters:
                    distance = self.getEuclideanDistance(self.irisdata[i][:-1], self.clusters[key]['mean'])
                    if (len(shortest_cluster) == 0):
                        shortest_cluster.update({"cluster": key, "distance": distance})
                    else:
                        if (shortest_cluster['distance'] > distance):
                            shortest_cluster['distance'] = distance
                            shortest_cluster['cluster'] = key
                            #     print ('key: ', key, 'dis: ', distance)
                # print ('####\nshortest: ', shortest_cluster['cluster'], 'dis: ', shortest_cluster['distance'])
                # put points to each cluster
                self.clusters[shortest_cluster['cluster']]['cluster_indexes'].append(i)
            # update mean
            for key in self.clusters:
                mean = np.array([])
                for index in self.clusters[key]['cluster_indexes']:
                    if len(mean) == 0:
                        mean = np.array(self.irisdata[index][:-1])
                    else:
                        mean += self.irisdata[index][:-1]
                mean /= len(self.clusters[key]['cluster_indexes'])
                self.clusters[key]['mean'] = mean.tolist()
            if (self.isConvergy()):
                break
            iterative += 1
        print ('Iterative: ', iterative)

    def isConvergy(self):
        if len(self.last_means) == 0:
            for key in self.clusters:
                self.last_means.update({key:self.clusters[key]['mean']})
            return False
        else:
            convergy_flag = 0
            for key in self.clusters:
                if (self.last_means[key] == self.clusters[key]['mean']):
                    convergy_flag += 1
            if (convergy_flag == len(self.last_means)):
                return True
            else:
                for key in self.clusters:
                    self.last_means.update({key:self.clusters[key]['mean']})
                return False

    def getEuclideanDistance(self, vector_a, vector_b):
        distance = 0
        for v1, v2 in zip(vector_a, vector_b):
            distance += math.pow(v1-v2, 2)
        return math.sqrt(distance)

    def getAccuracy(self, cluster):
        cluster_list = {}
        indexis = self.clusters[cluster]['cluster_indexes']
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

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    iris_kmean = IrisKMeans('iris_data_set/iris.data')
    init_index = iris_kmean.getFirstSampleIndex()
    iris_kmean.kMeansClustering(init_index, k=3)
    for key in iris_kmean.clusters:
        print ('cluster', key, ':')
        print ('\t(a) Cluster Center -->',np.array(iris_kmean.clusters[key]['mean']))
        print ('\t(b) Number of member in cluster -->',len(iris_kmean.clusters[key]['cluster_indexes']))
        cluster_list, num_wrong = iris_kmean.getAccuracy(key)
        print ('\t(c) Statistic -->', cluster_list)
        print ('\t    Number of Wrong -->', num_wrong)
