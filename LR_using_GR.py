# -*- coding: UTF-8 -*-
# Linear Regression using Gradient descent

'''
用梯度下降方法解决线性回归问题
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 构建数据
points_num = 100
vectors = []

# 用Numpy正态随机分布函数生成100个点
# 这些点的（x,y）坐标值对应线性方程y = 0.1 * x + 0.2
# weight = 0.1, bias = 0.2

for i in xrange(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors] # 真实的x， 输入
y_data = [v[1] for v in vectors] # 真实的y， 输入

# 图像1：展示100个随机数据点
plt.plot(x_data, y_data, 'r*', label="original data")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 初始化Weight
b = tf.Variable(tf.zeros([1])) # 初始化Bias
y = W * x_data + b # 模型计算出来的y

# 定义损失函数（loss/cost funciton）
# 对所有维度计算((y-y_data)^2)之和/N
loss = tf.reduce_mean(tf.square(y - y_data))

# 用梯度下降的优化器来优化损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5) # 设置学习率
train = optimizer.minimize(loss) # 最小化loss

# 初始化数据流图中的所有变量
init = tf.global_variables_initializer()
# 创建会话
with tf.Session() as sess:
    sess.run(init)
    # 训练20次
    for step in xrange(20):
        # 优化每一步
        sess.run(train)
        # 打印每一步的权重和偏差
        print( "Step=%d, Loss=%f, [Weight=%f Bias=%f]"
              % (step, sess.run(loss), sess.run(W), sess.run(b)) )

        # 图像2：绘制所有点并绘制出最佳拟合直线
    plt.plot(x_data, y_data, 'r*', label="original data")
    plt.title("Linear Regression using Gradient Descent")
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label = "Fitted Line")# 拟合的线
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
