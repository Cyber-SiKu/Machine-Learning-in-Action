# k-近邻算法

from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shap[0]

#count the distance
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
####################

    sortedDistIndicies = distance.argsort()

    classCount = {}
#select the less distance point
    for i in range(k):
        voteIlabek = labels[sortedDistIndicies[i]]
        classCount[voteIlabek] = classCount.get(voteIlabek,0) + 1
###############################

#sort
    sortClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True)
#####
    return sortClassCount[0][0]
