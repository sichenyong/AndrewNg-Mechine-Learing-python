import numpy as np
import matplotlib.pyplot as plt
def visualize_boundary(x_data, y_data, svm,xmin, xmax, ymin, ymax, message,figure_num):
    y_pos_index = np.where(y_data == 1)
    # print(y_pos_index)
    y_neg_index = np.where(y_data == 0)
    x_pos = x_data[y_pos_index[0], :]
    x_neg = x_data[y_neg_index[0], :]
    plt.figure(figure_num)
    plt.scatter(x_pos[:, 0], x_pos[:, 1],marker="+")
    plt.scatter(x_neg[:, 0], x_neg[:, 1])
    plt.legend(("positive data", "negative data"))
    plt.title(message)
    x_vals = np.linspace(xmin, xmax, 100)
    y_vals = np.linspace(ymin, ymax, 100)
    z_vals = np.zeros((len(y_vals),len(x_vals)))
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            z_vals[j][i] = svm.predict(np.array([[x_vals[i], y_vals[j]]]))
    plt.contour(x_vals, y_vals, z_vals, [0])
    plt.show()
