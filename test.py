import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

x_axis = range(1, 25)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("训练过程中的Loss值和准确率")
ax.set_ylabel("Loss值")
ax.set_xlabel("训练轮数")
ax.grid(which='minor', alpha=0.5)  # 设置网格
ax.grid(which='major', alpha=0.5)
ax.plot(x_axis, x_axis, 'r', label="train loss")
# ax.plot(x_axis, x_axis, 'g', label="train accuracy")
ax.legend()
plt.show()