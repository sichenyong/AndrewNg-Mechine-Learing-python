import matplotlib.pyplot as plt
import numpy as np

def plot_2D(x_data, y_data,figure_num):
    y_pos_index = np.where(y_data == 1)
    # print(y_pos_index)
    y_neg_index = np.where(y_data == 0)
    x_pos = x_data[y_pos_index[0], :]
    x_neg = x_data[y_neg_index[0], :]
    plt.figure(figure_num)
    plt.scatter(x_pos[:, 0], x_pos[:, 1],marker="+")
    plt.scatter(x_neg[:, 0], x_neg[:, 1])
    plt.legend(("positive data", "negative data"))
    plt.show()