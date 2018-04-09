# 3.4使用决策树预测隐形眼镜类型
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
import trees
lensesTree = trees.createTree(lenses, lensesLabels)
import treePlotter
treePlotter.createPlot(lensesTree)
