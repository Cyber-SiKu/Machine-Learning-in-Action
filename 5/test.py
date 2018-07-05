import svmMLiA
from numpy import shape, mat
from importlib import reload

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
labelArr

b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
alphas[alphas > 0]
# 支持响亮的个数
shape(alphas[alphas > 0])

# 显示哪些数据点是支持向量
for i in range(100):
    if alphas[i] > 0.0:
        print("dataAddr[%d]: " % i, dataArr[i], "labelArr[%d]: " % i, dataArr[i])

b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
ws

dataMat = mat(dataArr)
dataMat[0]*mat(ws)+b
labelArr[0]
dataMat[2]*mat(ws)+b


reload(svmMLiA)
svmMLiA.testRbf()