import random
from numpy import mat, nonzero, shape, zeros, multiply, exp, sign


# 6-1 SMO 算法的辅助函数


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineAddr = line.strip().split('\t')
        dataMat.append([float(lineAddr[0]), float(lineAddr[1])])
        labelMat.append(float(lineAddr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''
    返回 [0,m] 中一个非 i 的整数
    :param i:第一个alpha 的标
    :param m:所有alpha的数目
    :return:[0，m] 中一个非 i 的整数
    '''
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    '''
    调整大于H小于L的alpha值
    :param aj: alpha[j]
    :param H:上限
    :param L:下限
    :return:返回调整后的alpha
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 6-2 简化的 SMO 算法


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版SMO算法
    :param dataMatIn: 数据集
    :param classLabels:类标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 取消前最大循环次数
    :return:
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    print(m)
    alphas = mat(zeros((m, 1)))
    iter = 0  # 记录在没有任何 alpha 改变的情况下遍历数据集的次数
    while (iter < maxIter):  # 当 iter 达到输入值 maxIter 时，函数结束运行并退出
        alphaPairsChanged = 0  # 记录 alpha 是否已经进行优化
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T *
                        (dataMatrix * dataMatrix[i, :].T)) + b  # fXi 预测的类别
            Ei = fXi - float(labelMat[i])  # 计算误差
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei < toler) and
                     (alphas[i] > 0)):  # 判断误差是否过大
                # 如果 alpha 可以更改，进入优化过程
                j = selectJrand(i, m)  # 随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b  # fXj 计算第二个alpha值
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证 alpha 在 C 和 0 之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:  # 如果L和H相等则不做任何改变直接运行下一次for
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T  # eta 是 alpha[j] 的最优修改量
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not move enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * \
                    (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number : %d" % iter)
    return b, alphas


# 6-3 完整版的 Platt SMO 的支持函数
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 6-6 增加参数 kTup
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存 [i, j] 所以(self.m, 2)的格式
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        # 6-6 增加已下两行
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    '''
    计算E
    calculate the deviation
    :param oS: 缓存
    :param k: 下标
    :return: Ecanshu
    '''
    # 6-7 修改 fXk
    # fXk = float(multiply(oS.alphas, oS.labelMat).T*(oS.X*oS.X[k, :].T))+oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    '''
    选取有内循环的 alpha 值使得每次优化中采用最大步长
    select the value of j to take the mast step‘lenth in every step
    :param i: 外循环 alpha 的下标
    :param oS: 缓存
    :param Ei: 外循环的 alpha
    :return: 内循环的 alpha 的下标和 alpha
    '''
    # 内循环中的启发式方法
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 将输入值 Ei 在缓存中设置成有效的(已经计算好)
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 非零E值对应的 alpha 值
    if (len(validEcacheList)) > 1:
        # 选取使得改变最大的值
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # （以下两行）选择具有最大步长的 j
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 随机选取一个 alpha
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''
    计算误差并存入缓存
    :param oS: 缓存
    :param k: 要计算误差的下标
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 6-4 完整的 Platt SMO 算法中的优化例程
def innerL(i, oS):
    '''
    使用了数据结构 optStruct 使用selectJ() 而不是 selectJrand()来选择第二个 alpha
    :param i: 下标
    :param oS: 存储各种数据
    :return: alpha是（1）否（0）改变
    '''
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 第二个 alpha 选择使用启发式方法
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 6-7 修改 eta
        # eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - \
        #     oS.X[i, :]*oS.X[i, :].T - \
        #     oS.X[j, :]*oS.X[j, :].T
        eta = 2.0*oS.K[i, j]-oS.K[i, i]-oS.K[j, j]
        if eta >= 0:
            print("eta >=0 ")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新错误差值缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        # 更新错误差缓存
        updateEk(oS, i)
        # 6-7 修改 b1,b2
        # b1 = oS.b - Ei - \
        #     oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i, :]*oS.X[i, :].T - \
        #     oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i, :]*oS.X[j, :].T
        # b2 = oS.b - Ej - \
        #     oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
        #     oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0


# 6-5 完整版 Platt SMO 的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''

    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup:
    :return:
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True  # 标记是否需要遍历整个Set
    alphaPairschanged = 0
    while (iter < maxIter) and ((alphaPairschanged > 0) or (entireSet)):
        alphaPairschanged = 0
        if entireSet:
            # 遍历所有的值
            for i in range(oS.m):
                alphaPairschanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" %
                      (iter, i, alphaPairschanged))
            iter += 1
        else:
            # 遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairschanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" %
                      (iter, i, alphaPairschanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairschanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


# 计算 w
def calcWs(alphas, dataAddr, classLabels):
    X = mat(dataAddr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w


# 6-6 核转换函数
def kernelTrans(X, A, kTup):
    '''
    核转换函数
    :param X:
    :param A:
    :param kTup: 元组（核函数信息）第一个参数是描述所用核函数类型的字符串
    :return:
    '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        # 元素间除法
        K = exp(K / (-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That kernel is not recognized')
    return K
# 修改 optStruct

# 6-7 使用核函数时需要对innerL()和calcEk()函数进行修改

# 6-8 利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    # 构建支持向量矩阵
    sVs= datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is %f" % (float(errorCount)/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is %f" % (float(errorCount)/m))




# 6-9 基于 SVM 的手写數字识别
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

def inputTestDigits(kTup=('rbf', 10)):
    from os import listdir
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    # testFileList = listdir("testDigits")
    # filename = input("input the test:")
    # dataArr = img2vector(filename)
    # labelArr = 1
    # dataMat = mat(dataArr)
    # print(datMat)
    # kernelEval = kernelTrans(sVs, datMat, kTup)
    # predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
    # if sign(predict) != sign(labelArr):
    #     print("9")
    # else:
    #     print("1")
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    i = int(input("input i:"))
    while i != 0:
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        print("预测：%d" % sign(predict))
        print("真实：%d" % sign(labelArr[i]))
        i = int(input("input i:"))
