import random
from numpy import mat, nonzero, shape, zeros, multiply

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
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存 [i, j] 所以(self.m, 2)的格式
        self.eCache = mat(zeros((self.m, 2)))

    def calcEk(oS, k):
        fXk = float(multiply(oS.alphas, oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def selectJ(i, oS, Ei):
        # 内循环中的启发式方法
        maxK = -1; maxDeltaE = 0; Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = nonzero(oS.eCache[:, 0].A)[0] # nonzero() 返回 数组中非零元素的信息
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k ==i:
                    continue
                Ek = calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                # （以下两行）选择具有最大步长的j
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE;Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, oS.m)
            Ej = calcEk(oS, j)
        return j, Ej

    def updateEk(oS, k):
        Ek = calcEk(oS, k)
        oS.eCache[k] = [1, Ek]