# -*- coding: UTF-8 -*-

# 引入tensorflow
import tensorflow as tf

# 创建两个常量
const1 = tf.constant([[2,2]])
const2 = tf.constant([[4],[4]])

multiple = tf.matmul(const1, const2)

# 输出multiple
print multiple

# 创建了Session
sess = tf.Session()

# 运行矩阵乘法
result = sess.run(multiple)

# 打印
print result

if const1.graph is tf.get_default_graph():
    print "const1所在的图（Graph）是当前上下文默认的图"

# 关闭已用完的session
sess.close()

# 第二种创建与关闭session
with tf.Session() as sess:
    result2 = sess.run(multiple)
    print "Multiple的结果是 %s" % result2