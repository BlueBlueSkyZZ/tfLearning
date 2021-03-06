# -*- coding: UTF-8 -*-


import matplotlib.pyplot as plt
import numpy as np

# 创建数据,等分
x = np.linspace(-4, 4, 50)
# y = 3 * x + 4
y1 = 3 * x + 4
y2 = x ** 2

# 构建第一张图
plt.figure(num=1, figsize=(7,6))
plt.plot(x, y1)
plt.plot(x, y2, color="red", linewidth=3.0, linestyle="--")

# 构建第二张图
plt.figure(num=2)
plt.plot(x, y2, color="green")

# 显示图像
plt.show()
