# ---- QT 
import sys
import random

# ---- Matplotlib 
import numpy as np
import matplotlib.pyplot as plt


# y = []
# for i in range(100):
#     y.append(random.random())
#     plt.plot(y, label="Training error", color="orange")
#     plt.pause(0.001)
# plt.show()


# x = np.arange(12).reshape((3,4))
# print(x)

# sum = x.sum(axis=0)
# print(sum)


# a = np.array([0,0,0,0])
# b = np.array([5])

# c = np.concatenate([a,b], axis=0)
# print(c)

x = np.array([])
y = np.array([1,2])
# x = [[1,2,3], [5,6,7]]
# y = [[4], [8]]
c = np.concatenate([x,y], axis=0)
print(c)