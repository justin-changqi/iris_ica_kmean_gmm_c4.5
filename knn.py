import math
class Knn:
    def __init__(self):
        pass
    def kNearestNeighbors(self, train, test, k=3):
        correct = 0
        for data in test:
            if data[-1] == self.getVoteResult(train, data[:-1], k):
                correct += 1
        # return format(correct/len(test), '.3f')
        return correct/len(test)

    def getVoteResult(self, train, predict, k):
        distances = []
        for data in train:
            squre_sum = 0
            for i in range(len(data)-1):
                squre_sum += pow(data[i] - predict[i], 2)
            distances.append([math.sqrt(squre_sum), data[-1]])
        # print (distances)
        # sort with distance
        distances_sorted = []
        for ele in distances:
            if len(distances_sorted) == 0:
                distances_sorted.append(ele)
            else:
                index = 0
                for i in range(len(distances_sorted)):
                    if ele[0] > distances_sorted[index][0]:
                        index = i
                    else:
                        break
                distances_sorted.insert(index, ele)
        # print (np.array(distances_sorted))
        vote = []
        for data in distances_sorted[:k]:
            index = -1
            for i in range(len(vote)):
                if vote[i][0] == data[1]:
                    index = i
                    break
            if index == -1:
                vote.append([data[1], 1])
            else:
                vote[index][1] += 1
        max_class = [-1, -1]    # [class, number of vote]
        for data in vote:
            if data[1] > max_class[1]:
                max_class[0] = data[0]
                max_class[1] = data[1]
        return max_class[0]
