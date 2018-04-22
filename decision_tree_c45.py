import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree

# data source:  https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html

# outlook:
#     0: sunny
#     1: overcast
#     2: rain
# tempeature:
#     0: low
#     1: high
# humidity:
#     0: low
#     1: high
# windy:
#     0: false
#     1: true
# play:
#     0: don't play
#     1: play

class Node:
    def __init__(self, isLeaf, feature):
        self.feature = feature
        self.isLeaf = isLeaf
        self.children = []

class C45:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        self.feature_map = {'outlook':{'sunny':0, 'overcast':1, 'rain':2}, \
                            'tempeature':{'low':0, 'high':1}, \
                            'humidity':{'low':0, 'high':1}, \
                            'windy':{'false':0, 'true':1}, \
                            'play':{'nplay':0, 'paly':1}}
        self.feature_list = {'outlook': 0, 'tempeature':1, 'humidity':2, 'windy':3}
        self.train = np.array(df.astype(float).values.tolist())
        self.En_T = self.entropy(self.train.T[4])
        self.partitionData(self.getBestThreshold(self.train.T[2]), 2)
        self.partitionData(self.getBestThreshold(self.train.T[1]), 1)
        # print (self.train)
        root_indexes = list(range(len(self.train)))
        # root_feature = self.getMaxGainRate(self.feature_list, root_indexes)
        self.tree = self.generateTree(self.feature_list, root_indexes)
        self.printTree(self.tree)
        # print (self.slitIndexes('outlook', root_indexes))

    def getBestThreshold(self, data):
        # 1. Sort data
        sorted_data = np.sort(data)
        # 2. Remove redundancey
        delete_index = []
        for i in range(1, len(sorted_data)):
            if sorted_data[i] == sorted_data[i-1]:
                delete_index.append(i)
        sorted_data = np.delete(sorted_data, delete_index)
        # 3. Partitioned data with H0
        # 4. Compute all gain
        # 5. Return H0 with max gain
        max_gain = {'TH': 0, 'gain': 0}
        for h0 in sorted_data:
            partitioned_data = np.zeros(len(data))
            for j in range(len(data)):
                if (data[j] <= h0):
                    partitioned_data[j] = 0.
                else:
                    partitioned_data[j] = 1.
            gain = self.En_T - self.entropy(partitioned_data, self.train.T[4])
            if (max_gain['gain'] < gain):
                max_gain['TH'] = h0
                max_gain['gain'] = gain
        return max_gain['TH']


    def entropy(self, data_a, data_b=None):
        if (data_b is None):
            sorted_data = np.sort(data_a)
            ele_sum = [1.]
            for i in range(1, len(sorted_data)):
                if (sorted_data[i] == sorted_data[i-1]):
                    ele_sum[-1] += 1.
                else:
                    ele_sum.append(1.)
            probabilities = np.array(ele_sum) / len(data_a)
            # print (probabilities)
            entropy = 0
            for p in probabilities:
                entropy += -p*(math.log(p)/math.log(2))
            # print (entropy)
            return entropy
        else:
            partition_data = {}
            for i in range(len(data_a)):
                if (data_a[i] not in partition_data):
                    partition_data.update({data_a[i]:[i]})
                else:
                    partition_data[data_a[i]].append(i)
            # print (partition_data)
            entropy = 0
            for key in partition_data:
                p_key = len(partition_data[key]) / len(data_a)
                sub_partition = []
                for index in partition_data[key]:
                    sub_partition.append(data_b[index])
                entropy +=  p_key*self.entropy(sub_partition)
            # print (entropy)
            return entropy

    def partitionData(self, th, index):
        for i in range(len(self.train)):
            if (self.train[i][index] <= th):
                self.train[i][index] = 0
            else:
                self.train[i][index] = 1

    def getMaxGainRate(self, features, indexes):
        max_gain = {'feature': '', 'gain': -1}
        for key in features:
            data_vector = []
            for i in indexes:
                data_vector.append(self.train.T[features[key]][i])
            data_array = np.array(data_vector)
            gain = self.En_T - self.entropy(data_array, self.train.T[-1])
            split_gain = self.entropy(data_array)
            if split_gain == 0:
                gain_rate = 0
            else:
                gain_rate =  gain/split_gain
            if (max_gain['gain'] < gain_rate):
                max_gain['gain'] = gain_rate
                max_gain['feature'] = key
            # print (gain)
            # print ('key:', key, '-->', split_gain)
            # print ('key:', key, '-->', np.array(data_vector))
        # print (max_gain)
        return max_gain['feature']

    def getSplitProbability(self, feature_key, data_array):
        slit_p = np.zeros(len(self.feature_map[feature_key]))
        index = 0
        for key in self.feature_map[feature_key]:
            for data in data_array:
                if (data == self.feature_map[feature_key][key]):
                    slit_p[index] += 1.
            index += 1
        # print (slit_p / len(data_array))
        return (slit_p / len(data_array))

    def generateTree(self, feature_list, indexes):
        all_same = self.allSameResult(indexes)
        new_feature_list = feature_list
        if len(indexes) == 0:
            # print ('indexes = 0')
            return Node(True, None)
        elif (all_same is not False):
            # print ('all same')
            return Node(True, all_same)
        elif (len(feature_list) == 0):
            # print ('no feature_list')
            return Node(True, self.getMaxGainRate(feature_list, indexes))
        else:
            best = self.getMaxGainRate(feature_list, indexes)
            # print ('recursive: ', best)
            node = Node(False, best)
            slit_item = self.slitIndexes(best, indexes)
            del new_feature_list[best]
            for slit_indexes in slit_item:
                # print (slit_indexes)
                node.children.append(self.generateTree(new_feature_list, slit_indexes))
            return node

    def allSameResult(self, indexes):
        if (len(indexes) == 0):
            return False
        else:
            item = self.train.T[-1][indexes[0]]
            for index in indexes:
                if (self.train.T[-1][index] != item):
                    return False
            return item

    def slitIndexes(self, feature, indexes):
        # slit_indexes = {}
        slit_indexes = []
        feature_index = self.feature_list[feature]
        for key in self.feature_map[feature]:
            index = []
            for i in indexes:
                if (self.feature_map[feature][key] == self.train.T[feature_index][i]):
                    index.append(i)
            # slit_indexes.update({key:index})
            slit_indexes.append(index)
        return slit_indexes

    def printTree(self, tree):
        self.printNode(tree)

    def printNode(self, node, indent=''):
        if not node.isLeaf:
            print (indent, node.feature, ': ')
            for children in node.children:
                self.printNode(children, '    ')
        else:
            pass

if __name__ == '__main__':
    c45 = C45('playgolf.csv')
