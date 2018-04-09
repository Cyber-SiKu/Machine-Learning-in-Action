#!/usr/bin/env python3
# coding=utf-8

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
    xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction',
    va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('搜索', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('其他', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# 3-6 获得叶节点的数目和树的层数
# 前文 树用dict实现 dict第一个key 为根节点 其余节点组成新的dict为根节点的value 递归之
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 若不是叶节点 其属性必为dict
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDeth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDeth(secondDict[key]) #递归调用
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 输出预先存储的树的信息
def retrieveT(i):
    listOfTrees = [{
        'no surfacing': {
            0: 'no', 1: {
                'flippers': {
                    0: 'no', 1: 'yes'
                }
            }
        }
    }, {
        'no surfacing': {
            0: 'no', 1: {
                'flippers': {
                    0: {
                        'head': {
                            0: 'no', 1: 'yes'
                        }
                    }, 1: 'no'
                }
            }
        }
    }]
    return listOfTrees[i]

# 3.7 plotTree函数
# 在父子节点之间填充属性
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[0]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    # 计算宽与高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDeth(myTree)

    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff+(1.0 + float(numLeafs))/2.0/plotTree.totalw, \
                plotTree.yoff)
# 标记子节点的属性
    plotMidText(cntrPt, parentPt, nodeTxt)

    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
# 减少y的偏移
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalw
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
