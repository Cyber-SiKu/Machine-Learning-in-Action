#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy
import random
# ADD_1 程序清单4-2

# 4.1 词表到向量的转换函数

def loadDataSet():# 返回 词条切割后的文档 类别标签
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problem', 'help', 'please'],
                    ['myabe', 'not', 'take', 'him', \
                        'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmatonal', 'is', 'so', 'cute', \
                        'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
                        'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    '''
        1:侮辱性言论
        0:正常言论
    '''
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 返回包含在文件中出现的不重复的列表
def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集的合集 | 符号求两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):#词汇表&文档->文档向量
    # 创建一个与词汇表等长的向量
    # 其中所含元素都为0的向量 
    returnVec = [0]*len(vocabList)
    # 遍历文档中所有的单词,若出现词汇表中的单词
    # 输出文档中的对应值设为 1
    for word in inputSet:
        if word in vocabList:
            returnVec[list(vocabList).index(word)] = 1
        else:
            print('the word: %s is not my Vocabulart!' % word)
    return returnVec

# 程序清单4-2 朴素贝叶斯分类器训练函数
'''
输入参数:
    trainMatrix-文档矩阵    trainCategory-每篇文档类别构成的向量
返回值:
    p0Vect, p1Vect, pAbusive
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化概率
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0
    p0Num = numpy.ones(numWords); p1Num = numpy.ones(numWords)  # 防止最终概率为0而无法计算
    p0Denom = 2.0; p1Denom = 2.0    # 初始化分母
    for i in range(numTrainDocs):
        # 分别计算侮辱和非侮辱词汇的数量
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            #
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #对每个元素作除法
    # p1Vect = p1Num/p1Denom
    #p0Vect = p0Num/p0Denom
    p1Vect = numpy.log(p1Num/p1Denom)
    p0Vect = numpy.log(p0Num/p0Denom)    
    return p0Vect, p1Vect, pAbusive



# 4-3 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)   #元素相乘
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)
    if(p1 > p0):
        return 1

    else:
        return 0

def testingNB():
    # 载入数据
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(numpy.array(trainMat), numpy.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))


# 4-4 朴素贝叶斯词袋模型
def bagOfWords2VecNN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 4-5 文件解析及完整的垃圾邮件测试函数
'''
接受一个大字符串,将其解析为字符串列表,并去掉长度小于2的字符串
'''
def textPares(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]







'''
对贝叶斯垃圾邮件自动处理
'''
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        # 导入并解析文件
        wordList = textPares(open('email/spam/%d.txt' % i, errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textPares(open('email/ham/%d.txt' % i, errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        #
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    # 随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
        errorCount = 0
        for docIndex in testSet:
            wordVector = setOfWords2Vec(vocabList, docList)
            if classList[docIndex] == 1:
                print("垃圾邮件")
            else:
                print("非垃圾邮件")
            if classifyNB(numpy.array(wordVector), p0V, p1V, pSpam) != \
        classList[docIndex]:
                errorCount += 1
        print('the error rate is: ', float(errorCount)/len(testSet))