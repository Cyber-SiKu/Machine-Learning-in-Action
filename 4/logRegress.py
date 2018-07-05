#   Logistic    回归

#   5-1 Logistic 回归梯度上升优化算法
from math import exp
from numpy import mat, shape, ones, array, arange, random


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
    from numpy import exp
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
    xcord1 = []
    ycord1 = []
    xcord2 = []
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


#   5-3 随即梯度上升算法
def stochastic_gradent_ascent_0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


#   5-4 改进的随机梯度上升算法
def stochastic_gradent_ascent_1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        # 1 data_index = range(m) # 1
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001
            '''
            alpha 每次迭代时都调整
            虽然每次都在减小,但是有一个常数项0.001保证不为零
            当处理的问题是动态变化的时候,可以适当的加大常数项,来保证新的值得到更大的回归系数
            warrning：
            j:迭代次数,i:样本下标 当j << max(i)时,alpha不是严格下降(如何避免见模拟退火等其他优化算法)
            '''
            rand_index = int(random.uniform(0, len(data_index)))  # 随机选取更新
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del (data_index[rand_index])  # 3.x range 返回range对象不支持del
    return weights


#   5-5 Logistic 回归分类函数
def classify_vector(in_X, weights):
    prob = sigmoid(sum(in_X * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open("/media/siku/新加卷/Code/Machine-Learning-in-Action/4/horseColicTraining.txt")
    fr_test = open("/media/siku/新加卷/Code/Machine-Learning-in-Action/4/horseColicTest.txt")
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    training_weights = stochastic_gradent_ascent_1(array(training_set), training_labels, 500)
    error_count = 0
    num_test_vector = 0.0
    for line in fr_test.readlines():
        num_test_vector += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(array(line_arr), training_weights)) != \
                int(curr_line[21]):
            error_count += 1
    error_rate = (float(error_count)/num_test_vector)
    print("the error rate of this test is ", error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("after %d iterations the average error rate is : \
        %f" % (num_tests, error_sum/float(num_tests)))


def test():
    # import os
    # os.chdir('/media/siku/新加卷/Code/Machine-Learning-in-Action/4')
    # data, label = load_data_set()
    # weights = grad_ascent(data, label)
    # print("the best weights are:\n", weights)
    # print("figure is:")
    # plot_best_fit(weights.getA())
    print("multi_test")
    multi_test()


if __name__ == '__main__':
    test()
