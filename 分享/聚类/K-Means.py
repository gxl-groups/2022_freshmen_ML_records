#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
from random import sample
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris

# 导入鸢尾花数据集

# 数据集包含150个样本（行）
# 数据集包含4个属性（列）：，Petal Length,Petal Width,Sepal Length,Sepal Width
# 以二维数据为例 假设k=2，X为鸢尾花数据集前两个属性
iris = load_iris()
X = iris.data[:,0:2] # 取特征空间中的前两个属性 (150, 2)
#print(len(X))

# 绘制原始数据分布图
plt.scatter(X[:, 0], X[:, 1], c = "blue", marker='o', label='source')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()


# 从X中随机选择k个样本作为初始“簇中心”向量： μ(1),μ(2),...,,μ(k)
# 随机获得两个数据
k = 3  # 表示n个聚类
c = sample(X.tolist(),k) # x中，随机选取n个聚类中心
max_iter = 0 # 记录迭代次数
while max_iter < 6:
    # 簇分配过程  
    label = []
    for i in range(len(X)):
        min = 1000
        index = 0
        for j in range(k):
            dist = np.sqrt(np.sum(np.square(X[i] - c[j])))
            if dist < min:
                min = dist
                index = j
        label.append(index)

    
    # 移动聚类中心
    for j in range(k):
        sum = np.zeros(2)
        count = 0  # 统计不同类别的样本个数
        for i in range(len(X)):
            if label[i] == j:
                sum = sum + X[i]
                count = count + 1
        c[j] = sum / count
    #print("迭代次数：",max_iter,"------------c：",c)
    # 设置迭代次数
    max_iter = max_iter + 1

# print(np.array(label))
label_pred = np.array(label)
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()






