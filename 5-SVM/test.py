import svmMLiA
from numpy import shape

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
labelArr

b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
alphas[alphas>0]
# 支持响亮的个数
shape(alphas[alphas>0])

# 显示哪些数据点是支持向量
for i in range(100):
    if alphas[i] > 0.0 :
        print("dataAddr[%d]: " % i, dataArr[i], "labelArr[%d]: " % i, dataArr[i])