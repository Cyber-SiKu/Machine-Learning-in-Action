# Logistic 回归

# 5-1 Logistic 回归梯度上升优化算法
from math import exp
from numpy import mat, shape, ones, exp, array, arange


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # 1.0 是 X0 的值，这里是为了计算方便设置为1.0
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_X):
    return 1.0 / (1 + exp(-in_X))


def grad_ascent(data_mat_in, class_labels):
    # 转化成numpy矩阵的数据类型
    data_matrix = mat(data_mat_in)  # 样本数据集
    label_mat = mat(class_labels).transpose()  # 类别标签
    #
    m, n = shape(data_matrix)
    alpha = 0.001  # 移动的步长
    max_cycles = 500  # 迭代次数
    weights = ones((n, 1))
    for k in range(max_cycles):
        # 矩阵运算
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)  # 计算真实值和预测值之间的差值，由此确定移动的方向
        weights = weights + alpha * data_matrix.transpose() * error  # 根据移动的方向调整回归系数
        #
    return weights


#   5-2 画出决策边界
def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_array = array(data_mat)
    n = shape(data_array)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker='s')
    ax.scatter(xcord2, ycord2, s=30, c="green", marker='x')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


def test():
    import os
    os.chdir('/media/siku/新加卷/Code/Machine-Learning-in-Action/4')
    data, label = load_data_set()
    weights = grad_ascent(data, label)
    print("the best weights are:\n", weights)


if __name__ == '__main__':
    test()
