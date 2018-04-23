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
        self.threshold = None
        # self.children = []
        self.children = {}

class C45:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        self.feature_map = {'outlook':{'sunny':0, 'overcast':1, 'rain':2}, \
                            'temperature':{'low':0, 'high':1}, \
                            'humidity':{'low':0, 'high':1}, \
                            'windy':{'false':0, 'true':1}, \
                            'play':{'nplay':0, 'paly':1}}
        self.result_key = 'play'
        self.continuous_data_key = ['temperature', 'humidity']
        self.tree = self.generateTree(df)
        self.printTree(self.tree)

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

    def getBestThreshold(self, vector, result_vector):
        # 1. Sort data
        sorted_data = np.sort(vector)
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
            partitioned_data = np.zeros(len(vector))
            for j in range(len(vector)):
                if (vector[j] <= h0):
                    partitioned_data[j] = 0.
                else:
                    partitioned_data[j] = 1.
            gain = self.entropy(result_vector) - self.entropy(partitioned_data, result_vector)
            if (max_gain['gain'] < gain):
                max_gain['TH'] = h0
                max_gain['gain'] = gain
        # print (max_gain['TH'])
        return max_gain['TH']

    def partitionData(self, key, data, threshold):
        for i in data.index:
            if(data[key][i] <= threshold):
                data[key][i] = 0.
            else:
                data[key][i] = 1.

    def DiscreteData(self, data):
        need_discrete = []
        th_values = {}
        partitioned_data = data.copy()
        for key in partitioned_data:
            for con_key in self.continuous_data_key:
                if con_key == key:
                    need_discrete.append(con_key)
        if len(need_discrete)!=0:
            for key in need_discrete:
                # get max threhold
                th = self.getBestThreshold(np.array(partitioned_data[key].tolist()), \
                                           np.array(partitioned_data[self.result_key].tolist()))
                # partition belong th
                self.partitionData(key, partitioned_data, th)
                th_values.update({key:th})
        # print (partitioned_data)
        return partitioned_data, th_values

    def getMaxGainRate(self, data):
        max_gain = {'feature': '', 'gain': -1}
        result = np.array(data[self.result_key].tolist())
        data_rm_result = data.drop(self.result_key, axis=1)
        for key in data_rm_result:
            data_array = np.array(data_rm_result[key].tolist())
            gain = self.entropy(result) - self.entropy(data_array, result)
            split_gain = self.entropy(data_array)
            if split_gain == 0:
                gain_rate = -1
            else:
                gain_rate =  gain/split_gain
            if (max_gain['gain'] < gain_rate):
                max_gain['gain'] = gain_rate
                max_gain['feature'] = key
        # print (max_gain['feature'])
        return max_gain['feature']

    def slitData(self, best, data):
        slit_data = {}
        for map_key in self.feature_map[best]:
            slit_data.update({map_key:data.copy()})
            for index in slit_data[map_key].index:
                # print (index)
                if(slit_data[map_key][best][index] != self.feature_map[best][map_key]):
                    slit_data[map_key].drop(index, inplace=True)
            slit_data[map_key].drop(best, axis=1, inplace=True)
        return slit_data

    def allSameResult(self, data):
        data_list = data[self.result_key].tolist()
        if (len(data_list) == 0):
            return False
        else:
            item = data_list[0]
            for ele in data_list:
                if ele != item:
                    return False
            return item

    def generateTree(self, data):
        all_same = self.allSameResult(data)
        if data[self.result_key].count() == 0:
            # print ('indexes = 0')
            return Node(True, None)
        elif (all_same is not False):
            # print ('all same: ', all_same)
            return Node(True, all_same)
        else:
            # partition data
            discrete_data, th_values = self.DiscreteData(data)
            # find max gain item and slit
            maxgain_key = self.getMaxGainRate(discrete_data)
            # create node for each class
            node = Node(False, maxgain_key)
            print ('--------', maxgain_key, '--------')
            print (data)
            print (th_values)
            if(maxgain_key in th_values):
                node.threshold = th_values[maxgain_key]
            slit_item = self.slitData(maxgain_key, data)
            for key in slit_item:
                # print ('child ', key,': ', slit_item[key])
                node.children.update({key:self.generateTree(slit_item[key])})
            return node

    def printTree(self, tree):
        self.printNode(tree)

    def printNode(self, node, indent=''):
        if not node.isLeaf:
            print (indent, node.feature)
            if node.threshold != None:
                print (indent, node.threshold, ': ')
            for key in node.children:
                print (indent+'   ', key)
                self.printNode(node.children[key], indent+'    ')
        else:
            print (indent, '    ',  node.feature)
            pass

if __name__ == '__main__':
    c45 = C45('playgolf.csv')
