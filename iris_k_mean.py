import pandas as pd
import numpy as np
import math

class IrisKMeans:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
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
        
        pass

if __name__ == "__main__":
    iris_kmean = IrisKMeans('iris_data_set/iris.data')
    init_index = iris_kmean.getFirstSampleIndex()
    iris_kmean.kMeansClustering(init_index, k=3)
    pass
