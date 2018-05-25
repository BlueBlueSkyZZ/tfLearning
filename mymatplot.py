# -*- coding: UTF-8 -*-


import matplotlib.pyplot as plt
import numpy as np

# 创建数据,等分
x = np.linspace(-2, 2, 100)
# y = 3 * x + 4
y1 = 3 * x + 4
y2 = x ** 2

# 创建图像
# plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)

# 显示图像
plt.show()
