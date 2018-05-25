# -*- coding: UTF-8 -*-

# 引入
import tensorflow as tf

# 构造图的结构
# 用线性方程的例子
w = tf.Variable(2.0, dtype=tf.float32, name="Weight") # 权重
b = tf.Variable(1.0, dtype=tf.float32, name="Bias") # 偏差
x = tf.placeholder(dtype=tf.float32, name="Bias") # 输入
with tf.name_scope("Output"): # 输出的命名空间
    y = w * x + b # 输出

# 定义保存日志的路径
path = "./log"

# 创建初始化所有变量（Variable）的操作
init = tf.global_variables_initializer()

# 创建会话
with tf.Session() as sess:
    sess.run(init) # 实现初始化操作
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x: 3.0})
    print("y = %s" % result)   # 打印结果