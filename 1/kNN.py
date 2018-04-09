#!/usr/bin/env python
# coding=utf-8

from numpy import *
import operator
from os import listdir
####################################2-1####################################
###data###
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,1.0]])
    labels = ['A','A','B','B']
    return group, labels
##########

#######function############
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
#count the distance
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
####################

    sortedDistIndicies = distance.argsort()

    classCount = {}
#select the lest distance point
    for i in range(k):
        voteIlabek = labels[sortedDistIndicies[i]]
        classCount[voteIlabek] = classCount.get(voteIlabek,0) + 1
###############################

#sort
    sortClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True)
#####
    return sortClassCount[0][0]
#################################################

####################################2-2####################################
'''
將文本记录转化为 Numpy 的解析程序.
输入数据为： 训练样本矩阵 和 类标签向量
'''
def file2matrix(filename):
    fr = open(filename)

    # get the lines of filename
    arrayOLines = fr.readlines()
    numberOFLines = len(arrayOLines)

    # create the return's array
    returnMat = zeros((numberOFLines,3))

    classLabelVector = []
    index = 0

    #analysis filename to list
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1])) # 这里必须告诉Python解释器的列表的存储类型，否则解释器当字符春处理
        index += 1

    return returnMat,classLabelVector
#############################################


####################2-3######################
'''
归一化数据 使不同类型的数据权重相同 转化到[0,1]
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    #normDataSet = normDataSet/tile(ranges,(m,1))
    normDataSet /= tile(ranges,(m,1))
    return normDataSet, ranges, minVals
#############################################

####################2-4####################
'''
分类器针对约会网站的测试代码
保留10%的数据作为测试
'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                    datingLabels[numTestVecs:m],3)
        print('the No.%-3d classifierResult came back with:%d, the real answer is %d'\
             % (i+1, classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount+=1.0

    print('the total error rate is %f' % (errorCount/float(numTestVecs)))
###########################################

########################2-5########################
def classifyPerson():
    resultList = ['not at all','in small doses','in large does']
    print("percenttage of time spent playing video games?")
    percentTats = float(input())
    print("frequent filer miles earned per year?")
    ffMiles = float(input())
    print("liters of ice cream consumed per year?")
    iceCream = float(input())
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    intArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((intArr-minVals)/ranges,\
                    normMat,datingLabels,3)
    print("You will probably like this person:",\
          resultList[classifierResult-1])

###################################################

##############################2.3.1##############################
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


##################################################################
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    
    for i in range(m):
        # （以下三行）从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest, \
                                    trainingMat, hwLabels, 3)
        print("the classifier came back with %d, the real answer is :%d"\
             % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total rate is: %2.2f" % (errorCount/float(mTest)))
