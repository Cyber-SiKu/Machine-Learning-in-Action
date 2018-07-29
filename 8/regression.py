#!/usr/bin/python3

# 8-1 标准回归函数和数据导入函数
from typing import List

from numpy import *


def load_dataSet(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat: List[float] = []
    fr = open(file_name)
    for line in fr.readline():
        line_arrary = []
        current_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arrary.append(float(current_line))
        data_mat.append(line_arrary)
        label_mat.append(float(current_line[-1]))
    return data_mat, label_mat


def stan_regress(x_array, y_array):
    x_mat = mat(x_array)
    y_mat = mat(y_array).T
    xTx = x_mat.T * x_mat
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, can't do inverse")
        return
    ws = xTx.T * (x_mat.T * y_mat)
    return ws
