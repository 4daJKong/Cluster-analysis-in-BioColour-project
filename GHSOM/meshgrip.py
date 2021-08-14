import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# dataset = [[0,0],[0,1],[1,0],[1,1]]
# for item in dataset:
#     plt.scatter(x = item[0], y = item[1])
# plt.axhline(y = 0.5)
# plt.axvline(x = 0.5)

dataset_point = [[0, 1, 0], [1, 0, 1], [2, 1, 0], [3, 0, 1], [4, 0, 1], [5, 0, 0], [6, 1, 1], [7, 1, 0], [8, 1, 1], [9, 1, 0], [10, 1, 0], [11, 1, 0], [12, 1, 0], [13, 1, 0], [14, 1, 0], [15, 1, 0], [16, 0, 1], [17, 0, 1], [18, 1, 0], [19, 0, 1]]
dataset_point = np.array(dataset_point)
plt.scatter(x = dataset_point[:,1],y = dataset_point[:,2])
plt.show()
