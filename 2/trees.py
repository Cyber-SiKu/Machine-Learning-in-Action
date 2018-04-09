# /usr/bin/env python3
#

#          3-1             #


from math import log
# 载入log函数用于计算信息增益
import operator

'''
计算dataSet的信息增益(香农熵)
dataSet: [...,类别]
'''


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # 创建一个字典，键值为最后一列的数值
    # 为所有的可能分类（即所有属性）创建字典 （以下5行）
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            '''
            如果当前键值不存在，则拓展字典并将当前键值加入字典
            每个键值都记录了当前类别出现大次数
            '''
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1.0
        shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 创建数据


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 3.2 按照给定的特征划分数据#
'''
dataSet :   待划分的数据集
axis    :   划分数据集的特征
value   :   需要返回的特征值
'''


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 抽取符合特征的数据
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)

    return retDataSet
#########################################

# 3.3 选择最好的数据集划分方式#


def chooseBestFeatureTosplit(dataSet):
    '''
    选取数据集,计算得出最好的划分数据集的特征
    :param dataSet: 数据集(
        1. 必须列表元素构成的列表 所有的列表元素都要有相同的数据长度
        2. 数据的最后一列或者每个实例的最后一个元素是当前实例的标签
    )
    :return: (the best feature to split dataSet)划分数据集的最好特征
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 计算每一种类别的香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历数据集中的所有特征
        # 创建惟一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        # 便利当前特征中的所有唯一的属性值
        # 对每一个特征划分一次数据集
        # 然后计算数据集的新熵值
        # 并对所有唯一特征值得到的熵求值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # 计算最好的信息增量
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

##################


def majorityCnt(classList):
    '''
    使用分类名称的列表
    然后创建键值为classlist中的唯一的数据字典
    字典对象存储了classlist中的每个类标签出现的频率
    最后利用operator操作键值排序字典
    并返回出现最多的分类名称
    :param classList:
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.iteritems(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedClassCount[0][0]
###################################


# 3.4 创建树的函数代码
def createTree(dataSet, labels):
    '''
    两个输入参数
    :param dataSet:数据集
    :param labels: 标签列表
    :return:
    返回条件
        1. 所有的类别都相同
        2. 使用完了了所有的类别,此时选择出现次数最多的类别返回
    '''
    classList = [example[-1] for example in dataSet]
    # 类别相同的停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有的特征返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureTosplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc="0.8")
leafNode = dict(boxstyle='round4', fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodetxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodetxt, xy=parentPt,
                            xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fracion', va='center',
                            ha='center', bbox=nodeType, arrowprops=args)


def createPlot():
    fig = plt.figure(1, facecolor='red')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(U'決策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    #  将标签转化为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 使用pickle模块存储决策树


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
